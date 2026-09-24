"""Batterie-Speicher als Python-Wrapper um die BOPTEST-Umgebung — BOPTESTs
`bestest_hydronic_heat_pump`-Testfall hat selbst keine Batterie im Gebäudemodell, deshalb
wird der Ladezustand hier komplett unabhängig von BOPTEST mitgeführt.

Vier Dinge, die dieser Wrapper zur zugrunde liegenden BOPTEST-Umgebung hinzufügt:

1. **Batterie-Zustand** (`battery_soc`, State of Charge, 0..1): eigener Python-Zustand,
   Schritt für Schritt fortgeschrieben — BOPTEST weiß davon nichts.
2. **Erweiterter Aktionsraum**: zur Wärmepumpen-Aktion (`oveHeaPumY_u`, geht an BOPTEST) kommt
   eine zweite Aktion `battery_power_kw` hinzu (geht *nicht* an BOPTEST, positiv = laden,
   negativ = entladen).
3. **Erweiterte Beobachtung**: der aktuelle `battery_soc` wird an die BOPTEST-Beobachtung
   angehängt, damit der Agent seinen eigenen Ladezustand sieht.
4. **Neu berechnete Kosten**: der Strom, den die Wärmepumpe braucht, kommt entweder aus dem
   Netz (kostet den aktuellen Preis) oder aus der Batterie (kostet nichts extra — wurde schon
   beim Laden bezahlt). Ersetzt BOPTESTs eigenes `cost_tot` (das jede Wärmepumpen-Kilowattstunde
   pauschal zum Netzpreis abrechnet und die Batterie gar nicht kennt) durch eine eigene,
   batteriebewusste Kostenrechnung (`info['step_cost']`).

Die eigentliche Belohnung (Kosten, Komfort, Verschleiß, Restwert) entsteht nicht hier,
sondern zentral in logic/reward.py::RewardWrapper — dieser Wrapper liefert nur die Zahlen.
"""
import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

# Batterie-Kenndaten (typische Hausbatterie) — Konstruktor-Defaults, in logic/envs.py leicht
# änderbar, ohne diese Datei anzufassen.
CAPACITY_KWH = 10.0          # Nutzbare Speicherkapazität
MAX_POWER_KW = 3.0            # Maximale Lade-/Entladeleistung
CHARGE_EFFICIENCY = 0.95        # Verlust beim Laden
DISCHARGE_EFFICIENCY = 0.95      # Verlust beim Entladen
MIN_SOC = 0.05                    # Untere Schongrenze (nie ganz leer)
MAX_SOC = 0.95                     # Obere Schongrenze (nie ganz voll)
INITIAL_SOC = 0.5                   # Ladezustand zu Beginn jeder Episode

PRICE_POINT_BY_SCENARIO = {
    'constant': 'PriceElectricPowerConstant',
    'dynamic': 'PriceElectricPowerDynamic',
    'highly_dynamic': 'PriceElectricPowerHighlyDynamic',
}


class BatteryEnv(gym.Wrapper):
    """Umhüllt eine BoptestGymEnv (siehe logic/boptest_gym_env.py) mit einer selbst
    verwalteten Batterie. `env` muss genau eine BOPTEST-Aktion haben (hier: die
    Wärmepumpen-Modulation) — die Batterie-Aktion kommt als zweite, zusätzliche Dimension
    hinzu und wird nie an BOPTEST geschickt.
    """

    def __init__(self, env, capacity_kwh: float = CAPACITY_KWH,
                max_power_kw: float = MAX_POWER_KW, charge_efficiency: float = CHARGE_EFFICIENCY,
                discharge_efficiency: float = DISCHARGE_EFFICIENCY, min_soc: float = MIN_SOC,
                max_soc: float = MAX_SOC, initial_soc: float = INITIAL_SOC):
        super().__init__(env)
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.initial_soc = initial_soc

        base = env.unwrapped
        self._n_base_actions = len(base.actions)
        if self._n_base_actions != 1:
            raise ValueError('BatteryEnv erwartet genau eine zugrunde liegende BOPTEST-Aktion '
                             f'(die Wärmepumpe), gefunden: {self._n_base_actions}.')

        # 2. Aktionsraum erweitern: [Wärmepumpen-Aktion(en) von BOPTEST, Batterieleistung in kW]
        low = np.append(base.action_space.low, -self.max_power_kw).astype(np.float32)
        high = np.append(base.action_space.high, self.max_power_kw).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 3. Beobachtungsraum erweitern: Ladezustand (0..1) anhängen
        low = np.append(base.observation_space.low, self.min_soc).astype(np.float32)
        high = np.append(base.observation_space.high, self.max_soc).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.battery_soc = self.initial_soc
        self._prices = np.array([])
        self._step_idx = 0
        self.cum_cost = 0.0

    # ---------- Preis-Zeitreihe für die Episode vorab laden ----------

    def _load_price_series(self):
        """Preisverlauf für die ganze Episode auf einen Schlag abrufen — dasselbe Muster wie
        logic/watch.py::record(): BOPTESTs "Vorhersage" ist bei diesen Randbedingungen eine
        deterministische, im Testfall fest hinterlegte Zeitreihe, keine unsichere Prognose."""
        base = self.env.unwrapped
        price_point = PRICE_POINT_BY_SCENARIO.get(base.scenario.get('electricity_price'))
        if price_point is None or price_point not in base.all_predictive_vars:
            self._prices = None   # kein Preisszenario verfügbar -> Kosten bleiben 0
            return
        horizon = base.max_episode_length - base.step_period
        payload = requests.put(f'{base.url}/forecast/{base.testid}',
                               json={'point_names': [price_point], 'horizon': int(horizon),
                                    'interval': int(base.step_period)}).json()['payload']
        self._prices = np.asarray(payload[price_point], dtype=float)

    def _current_price(self) -> float:
        if self._prices is None or len(self._prices) == 0:
            return 0.0
        idx = min(self._step_idx, len(self._prices) - 1)
        return float(self._prices[idx])

    # ---------- Gymnasium-Interface ----------

    def reset(self, seed=None, options=None):
        base_obs, info = self.env.reset(seed=seed, options=options)
        self.battery_soc = self.initial_soc
        self._step_idx = 0
        self.cum_cost = 0.0
        self._load_price_series()
        info = dict(info, battery_soc=self.battery_soc, battery_energy_kwh=self.usable_energy_kwh())
        return self._extend_obs(base_obs), info

    def step(self, action):
        action = np.asarray(action, dtype=float)

        # RBC-Baseline (logic/baselines.py) übergibt eine leere Aktion, wenn BOPTESTs
        # eingebauter Regler die Wärmepumpe steuert — dann bleibt die Batterie einfach
        # untätig (kein battery_power vorgegeben), statt einen Index-Fehler auszulösen.
        n_hp = self._n_base_actions if len(action) >= self._n_base_actions else 0
        hp_action = action[:n_hp]
        requested_power_kw = float(action[n_hp]) if len(action) > n_hp else 0.0

        actual_power_kw = self._apply_battery(requested_power_kw)

        base_obs, _base_reward, terminated, truncated, info = self.env.step(hp_action)
        res = info.get('res', {})
        heat_pump_power_kw = float(res.get('reaPHeaPum_y', 0.0)) / 1000.0

        # 4. Kosten: was die Wärmepumpe braucht, kommt aus dem Netz (kostet den aktuellen
        # Preis) oder aus der Batterie (kostet nichts extra, wurde beim Laden schon bezahlt).
        # Netzbezug = Wärmepumpe + Batterieladung (falls geladen wird) - Batterieentladung
        # (falls entladen wird), nach unten bei 0 gedeckelt (kein Einspeise-Erlös in diesem
        # einfachen Modell, siehe workbench.md).
        dt_hours = self.env.unwrapped.step_period / 3600.0
        grid_power_kw = max(0.0, heat_pump_power_kw + actual_power_kw)
        step_cost = grid_power_kw * self._current_price() * dt_hours
        self.cum_cost += step_cost

        price = self._current_price()
        self._step_idx += 1
        info = dict(info, battery_soc=self.battery_soc, battery_energy_kwh=self.usable_energy_kwh(),
                   battery_power_kw=actual_power_kw, heat_pump_power_kw=heat_pump_power_kw,
                   grid_power_kw=grid_power_kw, step_cost=step_cost, price=price)
        # Nur Kosten — die vollständige Belohnung setzt logic/reward.py::RewardWrapper.
        return self._extend_obs(base_obs), -step_cost, terminated, truncated, info

    # ---------- Batterie-Physik ----------

    def _apply_battery(self, requested_power_kw: float) -> float:
        """Setzt die angeforderte Batterieleistung so weit um, wie es der aktuelle
        Ladezustand physikalisch zulässt (nicht über MAX_SOC laden, nicht unter MIN_SOC
        entladen), und schreibt battery_soc entsprechend fort. Gibt die tatsächlich
        umgesetzte Leistung zurück (kW, positiv = laden, negativ = entladen)."""
        dt_hours = self.env.unwrapped.step_period / 3600.0
        requested_power_kw = float(np.clip(requested_power_kw, -self.max_power_kw, self.max_power_kw))

        if requested_power_kw >= 0:   # laden
            max_chargeable_kwh = (self.max_soc - self.battery_soc) * self.capacity_kwh
            max_chargeable_kw = max(0.0, max_chargeable_kwh / (dt_hours * self.charge_efficiency))
            actual_power_kw = min(requested_power_kw, max_chargeable_kw)
            self.battery_soc += actual_power_kw * dt_hours * self.charge_efficiency / self.capacity_kwh
        else:                          # entladen
            max_dischargeable_kwh = (self.battery_soc - self.min_soc) * self.capacity_kwh
            max_dischargeable_kw = max(0.0, max_dischargeable_kwh * self.discharge_efficiency / dt_hours)
            actual_power_kw = -min(-requested_power_kw, max_dischargeable_kw)
            self.battery_soc += actual_power_kw * dt_hours / (self.discharge_efficiency * self.capacity_kwh)

        self.battery_soc = float(np.clip(self.battery_soc, self.min_soc, self.max_soc))
        return actual_power_kw

    def usable_energy_kwh(self) -> float:
        """Energie, die sich noch aus der Batterie entnehmen ließe (über MIN_SOC, nach
        Entladeverlust) — Grundlage für den Restwert in logic/reward.py."""
        return (self.battery_soc - self.min_soc) * self.capacity_kwh * self.discharge_efficiency

    def _extend_obs(self, base_obs: np.ndarray) -> np.ndarray:
        return np.append(np.asarray(base_obs, dtype=np.float32), np.float32(self.battery_soc))
