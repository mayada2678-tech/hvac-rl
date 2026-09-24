"""Belohnungsfunktion des Agenten — an genau einer Stelle, als äußerster Wrapper um den
Umgebungs-Stapel aus logic/envs.py::make_env() (BoptestGymEnv -> BatteryEnv -> SolarEnv).
Die inneren Wrapper rechnen nur Physik und Kosten und legen sie ins info-Dict; welche dieser
Größen wie stark belohnt/bestraft werden, entscheidet ausschließlich diese Datei.

Belohnung je Schritt (alle Terme in Euro-Äquivalent, negativ = schlecht):

    r = -scale * ( Kosten
                 + w_comfort * Komfort-Defizit
                 + w_battery * Batterie-Durchsatz )
        + scale * w_terminal * Restwert-Änderung der Batterie      (nur im letzten Schritt)

1. **Kosten** (€): Netzbezug × aktueller Strompreis, schon abzüglich Batterie-Entladung und
   PV-Erzeugung (`info['step_cost']` von BatteryEnv/SolarEnv). Ohne Batterie-Schicht
   (`battery=False`) ersatzweise der Anstieg von BOPTESTs eigenem `cost_tot`.
2. **Komfort-Defizit** (K·h): Anstieg von BOPTESTs offiziellem KPI `tdis_tot` in diesem
   Schritt — dieselbe Größe, mit der BOPTEST am Ende bewertet. `w_comfort` (€ je K·h) legt
   fest, wie viel dem Agenten ein Kelvin-Stunde Komfortverletzung "wert" ist.
3. **Batterie-Verschleiß** (€): |Batterieleistung| × Schrittdauer × `w_battery` (€ je kWh
   Durchsatz). Ohne diesen Term ist Hin-und-her-Zyklieren bei kleinen Preisunterschieden
   gratis; mit ihm lohnt sich Laden/Entladen nur, wenn die Preisdifferenz den Verschleiß
   übersteigt.
4. **Batterie-Restwert** (€, nur am Episodenende): Die Batterie startet jede Episode mit
   Ladung (INITIAL_SOC), und Entladen kostet in der Kostenrechnung nichts — ohne Ausgleich
   könnte der Agent die Startladung einfach verbrauchen und bekäme diese Energie geschenkt.
   Deshalb wird am Ende die Änderung der entnehmbaren Energie gegenüber dem Start mit dem
   mittleren Strompreis der Episode bewertet: leerer als am Anfang = Abzug, voller = Gutschrift.

Die einzelnen Terme landen zusätzlich im info-Dict (`reward_cost`, `reward_comfort`,
`reward_battery`, `reward_terminal`, `discomfort_kh`), damit man in logic/watch.py bzw. der
Oberfläche sieht, *woraus* sich die Belohnung zusammensetzt.
"""
import gymnasium as gym

# Standardgewichte — in der Oberfläche bzw. configs/*.yaml (Schlüssel `reward`) überschreibbar.
W_COMFORT = 1.0      # € je Kelvin-Stunde Komfort-Defizit
W_BATTERY = 0.02      # € je kWh Batterie-Durchsatz (Laden + Entladen), typ. Verschleißkosten
W_TERMINAL = 1.0       # 1 = Restwert voll anrechnen, 0 = Restwert ignorieren (alter Stand)
SCALE = 1.0             # Gesamtskalierung (für die Lernrate relevant, nicht fürs Optimum)


class RewardWrapper(gym.Wrapper):
    """Äußerster Wrapper: ersetzt die Belohnung der inneren Schichten durch die hier
    definierte Zielfunktion (siehe Modul-Docstring)."""

    def __init__(self, env, w_comfort: float = W_COMFORT, w_battery: float = W_BATTERY,
                 w_terminal: float = W_TERMINAL, scale: float = SCALE):
        super().__init__(env)
        self.w_comfort = w_comfort
        self.w_battery = w_battery
        self.w_terminal = w_terminal
        self.scale = scale
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
        return obs, info

    def step(self, action):
        obs, _inner_reward, terminated, truncated, info = self.env.step(action)
        kpis = self.env.unwrapped.get_kpis()
        dt_hours = self.env.unwrapped.step_period / 3600.0

        # 1. Kosten
        if 'step_cost' in info:
            cost = float(info['step_cost'])
        else:   # reine Wärmepumpen-Umgebung ohne Batterie-Schicht
            cost_tot = float(kpis.get('cost_tot') or 0.0)
            cost = cost_tot - self._prev_cost_tot
            self._prev_cost_tot = cost_tot

        # 2. Komfort-Defizit
        tdis = float(kpis.get('tdis_tot') or 0.0)
        discomfort_kh = tdis - self._prev_tdis
        self._prev_tdis = tdis

        # 3. Batterie-Verschleiß
        throughput_kwh = abs(float(info.get('battery_power_kw', 0.0))) * dt_hours

        r_cost = -cost
        r_comfort = -self.w_comfort * discomfort_kh
        r_battery = -self.w_battery * throughput_kwh

        # 4. Batterie-Restwert am Episodenende
        self._price_sum += float(info.get('price', 0.0))
        self._n_steps += 1
        r_terminal = 0.0
        if (terminated or truncated) and self._initial_energy_kwh is not None:
            mean_price = self._price_sum / self._n_steps
            delta_kwh = float(info.get('battery_energy_kwh', self._initial_energy_kwh)) - self._initial_energy_kwh
            r_terminal = self.w_terminal * delta_kwh * mean_price

        reward = self.scale * (r_cost + r_comfort + r_battery + r_terminal)
        info = dict(info, reward_cost=r_cost, reward_comfort=r_comfort, reward_battery=r_battery,
                   reward_terminal=r_terminal, discomfort_kh=discomfort_kh)
        return obs, float(reward), terminated, truncated, info
