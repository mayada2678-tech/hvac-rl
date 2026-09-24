"""Photovoltaik-Erzeugung als Python-Wrapper um die (batteriefähige) BOPTEST-Umgebung —
dasselbe Prinzip wie logic/battery_env.py: BOPTEST simuliert nur das Gebäude, eine PV-Anlage
gibt es im Testfall nicht, deshalb wird sie hier komplett unabhängig berechnet.

Formel (anerkannte, vereinfachte PV-Ertragsrechnung ohne Verschattung/Temperaturderating):

    P_solar = HDirNor * A_panel * eta_panel / 1000        [kW]

- HDirNor: direkte Solarstrahlung, aus BOPTESTs Messgröße `weaSta_reaWeaHDirNor_y` (W/m²) —
  der tatsächlich realisierte Wert je Schritt, nicht die Vorhersage, aus demselben `res`-Dict,
  das schon logic/battery_env.py für die Wärmepumpenleistung nutzt.
- A_panel: angenommene Panelfläche (m²) — **eigene Annahme dieses Projekts, keine
  BOPTEST-Vorgabe**, siehe PANEL_AREA_M2 unten.
- eta_panel: Modulwirkungsgrad — 0.19 (19 %), typischer Bereich heutiger monokristalliner
  Module ist 18-20 %.
- /1000: W -> kW.

Wie bei der Batterie ist erzeugter Solarstrom "kostenlos": deckt er einen Teil des
Netzbezugs, den Wärmepumpe (und ggf. Batterieladung) sonst bräuchten, sinken die Kosten
entsprechend. Überschüssiger Solarstrom (mehr, als gerade gebraucht wird) wird in diesem
einfachen Modell nicht eingespeist/vergütet, sondern verworfen (Abregelung) — siehe
workbench.md für Erweiterungsideen (z. B. Überschuss zuerst die Batterie laden lassen).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Eigene Annahmen dieses Projekts (keine BOPTEST-Vorgabe) — hier zentral, leicht änderbar.
PANEL_AREA_M2 = 20.0        # Angenommene PV-Panelfläche
PANEL_EFFICIENCY = 0.19      # Modulwirkungsgrad (19 %)
HDIRNOR_MAX = 862.0           # Obergrenze für die Beobachtungs-Skalierung (siehe
                              # logic/envs.py::OBSERVATIONS, dort dieselbe Grenze für HDirNor)


class SolarEnv(gym.Wrapper):
    """Umhüllt eine (idealerweise batteriefähige, siehe logic/battery_env.py) BOPTEST-Umgebung
    mit einer PV-Anlage. Keine eigene Aktion — Solarerzeugung ist nicht steuerbar — sondern
    ergänzt nur die Beobachtung um die aktuelle Erzeugung und korrigiert die vom inneren
    Wrapper berechneten Kosten um den Anteil, den die PV-Anlage kostenlos deckt.
    """

    def __init__(self, env, panel_area_m2: float = PANEL_AREA_M2, panel_efficiency: float = PANEL_EFFICIENCY):
        super().__init__(env)
        self.panel_area_m2 = panel_area_m2
        self.panel_efficiency = panel_efficiency
        self.max_solar_kw = HDIRNOR_MAX * self.panel_area_m2 * self.panel_efficiency / 1000.0

        low = np.append(env.observation_space.low, 0.0).astype(np.float32)
        high = np.append(env.observation_space.high, self.max_solar_kw).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.cum_solar_savings = 0.0
        self.cum_cost = 0.0

    def _solar_power_kw(self, res: dict) -> float:
        h_dir_nor = float(res.get('weaSta_reaWeaHDirNor_y', 0.0) or 0.0)
        return h_dir_nor * self.panel_area_m2 * self.panel_efficiency / 1000.0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.cum_solar_savings = 0.0
        self.cum_cost = getattr(self.env, 'cum_cost', 0.0)
        info = dict(info, solar_power_kw=0.0)
        return self._extend_obs(obs, 0.0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        solar_kw = self._solar_power_kw(info.get('res', {}))

        # Solarstrom deckt zuerst den Netzbezug, der laut innerem Wrapper sonst nötig wäre
        # (Wärmepumpe + ggf. Batterieladung, schon abzüglich Batterieentladung) — kostenlos,
        # genau wie bereits aus der Batterie entnommener Strom.
        grid_power_kw = float(info.get('grid_power_kw', 0.0))
        price = float(info.get('price', 0.0))
        dt_hours = self.env.unwrapped.step_period / 3600.0
        offset_kw = min(solar_kw, grid_power_kw)
        savings = offset_kw * price * dt_hours
        self.cum_solar_savings += savings
        self.cum_cost = getattr(self.env, 'cum_cost', self.cum_cost) - self.cum_solar_savings

        info = dict(info, solar_power_kw=solar_kw, grid_power_kw=grid_power_kw - offset_kw,
                   step_cost=max(0.0, float(info.get('step_cost', 0.0)) - savings))
        # Nur Kosten — die vollständige Belohnung setzt logic/reward.py::RewardWrapper.
        return self._extend_obs(obs, solar_kw), reward + savings, terminated, truncated, info

    def _extend_obs(self, obs: np.ndarray, solar_kw: float) -> np.ndarray:
        return np.append(np.asarray(obs, dtype=np.float32), np.float32(solar_kw))
