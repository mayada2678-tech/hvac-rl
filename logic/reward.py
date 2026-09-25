"""Belohnungsfunktion des Agenten — an genau einer Stelle, als äußerster Wrapper um den
Umgebungs-Stapel aus logic/envs.py::make_env() (BoptestGymEnv -> BatteryEnv -> SolarEnv).
Die inneren Wrapper rechnen nur Physik und Kosten und legen sie ins info-Dict; welche dieser
Größen wie stark belohnt/bestraft werden, entscheidet ausschließlich diese Datei. Jeder
Parameter ist in der Oberfläche einstellbar (gui/reward_form.py).

Belohnung je Schritt — in BOPTESTs KPI-Einheiten (Kosten in € je m² Wohnfläche, Komfort in
Kelvin-Stunden), damit `w_comfort` dieselbe Bedeutung hat wie in BOPTEST-Gym:

    r = -scale * ( (Kosten + Verschleiß - Restwert) / Wohnfläche  +  w_comfort * ΔKomfort )

1. **Kosten** (€): Netzbezug × aktueller Strompreis, schon abzüglich Batterie-Entladung und
   PV-Erzeugung (`info['step_cost']` von BatteryEnv/SolarEnv). Ohne Batterie-Schicht
   (`battery=False`) ersatzweise der Anstieg von BOPTESTs eigenem `cost_tot`.
2. **ΔKomfort** (K·h): Anstieg von BOPTESTs offiziellem KPI `tdis_tot` in diesem Schritt.
3. **Verschleiß** (€): |Batterieleistung| × Schrittdauer × Verschleißkosten je kWh Durchsatz.
   Die Verschleißkosten folgen aus den Batterie-Kenndaten: ein Vollzyklus bewegt
   2 × Entladetiefe × Kapazität (einmal laden, einmal entladen), über die Lebensdauer also
   2 × Zyklen × Entladetiefe × Kapazität — verteilt auf den Anschaffungspreis:

       € je kWh Durchsatz = Anschaffungspreis (€/kWh) / (2 × Zyklen × Entladetiefe)

4. **Restwert** (€, nur im letzten Schritt): Die Batterie startet jede Episode mit Ladung,
   und Entladen kostet in der Kostenrechnung nichts — ohne Ausgleich könnte der Agent die
   Startladung verbrauchen und bekäme sie geschenkt. Deshalb wird am Ende die Änderung der
   entnehmbaren Energie gegenüber dem Start mit dem mittleren Strompreis der Episode
   bewertet (× `w_terminal`): leerer als am Anfang = Abzug, voller = Gutschrift.

Die einzelnen Anteile (in denselben Einheiten wie r, zusammen = r) landen zusätzlich im
info-Dict (`reward_cost`, `reward_comfort`, `reward_battery`, `reward_terminal`) und in den
Aufzeichnungen von logic/watch.py.
"""
import gymnasium as gym

# Namen der Einzelanteile im info-Dict (zusammen = r) — auch Spalten in logic/watch.py und in
# der Lernkurven-CSV von logic/training.py.
REWARD_PARTS = ['reward_cost', 'reward_comfort', 'reward_battery', 'reward_terminal']

# Standardwerte = beste Schätzung aus Literatur/Datenblättern, Quellen je Parameter in
# PARAM_INFO unten (dort auch Bereich und Beschriftung für die Oberfläche).
REWARD_DEFAULTS = {
    'w_comfort': 1.0,
    'floor_area_m2': 192.0,
    'battery_invest_eur_per_kwh': 500.0,
    'battery_cycle_life': 6000,
    'w_terminal': 1.0,
    'scale': 1.0,
}

# Komfort-Voreinstellungen genau wie BOPTEST-Gyms Beispielklassen
# (BoptestGymEnvRewardWeightCost / BoptestGymEnv / BoptestGymEnvRewardWeightDiscomfort).
COMFORT_PRESETS = {
    'Kostenbetont (BOPTEST-Gym, w = 0.1)': 0.1,
    'Ausgewogen (BOPTEST-Gym-Standard, w = 1)': 1.0,
    'Komfortbetont (BOPTEST-Gym, w = 10)': 10.0,
}

# Für die Oberfläche: Beschriftung, Bereich, Schrittweite, Nachkommastellen, Erklärung/Quelle.
PARAM_INFO = {
    'w_comfort': dict(
        label='Komfortgewicht w (€/m² je K·h)', lo=0.0, hi=1000.0, step=0.1, decimals=2,
        help='Wie teuer eine Kelvin-Stunde Komfortverletzung im Verhältnis zu den Stromkosten '
             'je m² Wohnfläche ist. Standard 1 = BOPTEST-Gym-Referenz (Arroyo et al., '
             'ibpsa/project1-boptest-gym, boptestGymEnv.py: cost_tot + 1·tdis_tot); dort '
             'außerdem 0.1 (kostenbetont) und 10 (komfortbetont).'),
    'floor_area_m2': dict(
        label='Wohnfläche (m²)', lo=1.0, hi=100_000.0, step=1.0, decimals=0,
        help='Rechnet Euro in BOPTESTs Kosten-KPI-Einheit €/m² um. 192 m² = Grundriss '
             '12 m × 16 m laut BOPTEST-Dokumentation von bestest_hydronic_heat_pump. Nur bei '
             'einem anderen Testfall ändern.'),
    'battery_invest_eur_per_kwh': dict(
        label='Batterie-Anschaffungspreis (€/kWh)', lo=0.0, hi=5000.0, step=50.0, decimals=0,
        help='Grundlage der Verschleißkosten. 500 €/kWh ≈ typischer Preis eines '
             'LFP-Heimspeichers 2024/25 (Richtwert, grob 400-800 €/kWh inkl. Installation). '
             '0 = Verschleiß ignorieren.'),
    'battery_cycle_life': dict(
        label='Batterie-Lebensdauer (Vollzyklen)', lo=100, hi=50_000, step=500, decimals=0,
        help='Anzahl Vollzyklen bis zum Lebensende. 6000 = übliche Herstellerangabe/Garantie '
             'für LFP-Heimspeicher. Kalendarische Alterung ist nicht berücksichtigt.'),
    'w_terminal': dict(
        label='Batterie-Restwert am Episodenende (0-1)', lo=0.0, hi=1.0, step=0.25, decimals=2,
        help='1 = entnehmbare Restenergie am Ende voll zum mittleren Episodenpreis bewerten '
             '(ökonomisch korrekt: verhindert, dass der Agent die Startladung gratis '
             'verbraucht). 0 = Restwert ignorieren.'),
    'scale': dict(
        label='Gesamtskalierung der Belohnung', lo=0.001, hi=1000.0, step=0.5, decimals=3,
        help='Multipliziert die ganze Belohnung. Ändert nicht, was optimal ist, nur die '
             'Größenordnung der Gradienten. 1 = unverändert (SAC passt seine '
             'Entropie-Gewichtung ohnehin automatisch an).'),
}


def wear_cost_per_kwh(invest_eur_per_kwh: float, cycle_life: float, depth_of_discharge: float) -> float:
    """Verschleißkosten je kWh Batterie-Durchsatz (Laden + Entladen), siehe Modul-Docstring."""
    if cycle_life <= 0 or depth_of_discharge <= 0:
        return 0.0
    return invest_eur_per_kwh / (2.0 * cycle_life * depth_of_discharge)


class RewardWrapper(gym.Wrapper):
    """Äußerster Wrapper: ersetzt die Belohnung der inneren Schichten durch die hier
    definierte Zielfunktion (siehe Modul-Docstring)."""

    def __init__(self, env, w_comfort: float = REWARD_DEFAULTS['w_comfort'],
                 floor_area_m2: float = REWARD_DEFAULTS['floor_area_m2'],
                 battery_invest_eur_per_kwh: float = REWARD_DEFAULTS['battery_invest_eur_per_kwh'],
                 battery_cycle_life: float = REWARD_DEFAULTS['battery_cycle_life'],
                 w_terminal: float = REWARD_DEFAULTS['w_terminal'],
                 scale: float = REWARD_DEFAULTS['scale'],
                 w_battery: float | None = None):
        """w_battery: Verschleißkosten (€ je kWh Durchsatz) direkt vorgeben statt aus
        Anschaffungspreis/Zyklen/Entladetiefe abzuleiten (ältere Laufkonfigurationen)."""
        super().__init__(env)
        self.w_comfort = w_comfort
        self.floor_area_m2 = floor_area_m2
        self.battery_invest_eur_per_kwh = battery_invest_eur_per_kwh
        self.battery_cycle_life = battery_cycle_life
        self.w_terminal = w_terminal
        self.scale = scale
        self._w_battery_override = w_battery
        self.w_battery = w_battery or 0.0
        self._reset_state()

    def _reset_state(self):
        self._prev_tdis = 0.0
        self._prev_cost_tot = 0.0
        self._initial_energy_kwh = None
        self._price_sum = 0.0
        self._n_steps = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_state()
        self._initial_energy_kwh = info.get('battery_energy_kwh')
        if self._w_battery_override is None:
            self.w_battery = wear_cost_per_kwh(self.battery_invest_eur_per_kwh, self.battery_cycle_life,
                                               info.get('battery_dod', 0.0))
        return obs, info

    def step(self, action):
        obs, _inner_reward, terminated, truncated, info = self.env.step(action)
        kpis = self.env.unwrapped.get_kpis()
        dt_hours = self.env.unwrapped.step_period / 3600.0

        # 1. Kosten (€)
        if 'step_cost' in info:
            cost_eur = float(info['step_cost'])
        else:   # reine Wärmepumpen-Umgebung ohne Batterie-Schicht: BOPTEST rechnet in €/m²
            cost_tot = float(kpis.get('cost_tot') or 0.0)
            cost_eur = (cost_tot - self._prev_cost_tot) * self.floor_area_m2
            self._prev_cost_tot = cost_tot

        # 2. Komfort-Defizit (K·h)
        tdis = float(kpis.get('tdis_tot') or 0.0)
        discomfort_kh = tdis - self._prev_tdis
        self._prev_tdis = tdis

        # 3. Verschleiß (€)
        wear_eur = abs(float(info.get('battery_power_kw', 0.0))) * dt_hours * self.w_battery

        # 4. Restwert (€), nur im letzten Schritt
        self._price_sum += float(info.get('price', 0.0))
        self._n_steps += 1
        terminal_eur = 0.0
        if (terminated or truncated) and self._initial_energy_kwh is not None:
            mean_price = self._price_sum / self._n_steps
            delta_kwh = float(info.get('battery_energy_kwh', self._initial_energy_kwh)) - self._initial_energy_kwh
            terminal_eur = self.w_terminal * delta_kwh * mean_price

        per_m2 = self.scale / self.floor_area_m2
        parts = {'reward_cost': -cost_eur * per_m2,
                 'reward_comfort': -self.scale * self.w_comfort * discomfort_kh,
                 'reward_battery': -wear_eur * per_m2,
                 'reward_terminal': terminal_eur * per_m2}
        reward = sum(parts.values())
        info = dict(info, **parts, discomfort_kh=discomfort_kh, wear_eur=wear_eur)
        return obs, float(reward), terminated, truncated, info
