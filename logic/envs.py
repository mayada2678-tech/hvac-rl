"""BOPTEST-Umgebung aufbauen: Testfall `bestest_hydronic_heat_pump` (Einzonen-Wohngebäude mit
Wärmepumpe + Fußbodenheizung), dynamischer Strompreis, Komfort-Belohnung, plus zwei selbst
verwaltete Python-Schichten obendrauf, die der Testfall selbst nicht kennt:
  - eine Batterie (logic/battery_env.py) — zweite Aktion, eigener Ladezustand,
  - eine PV-Anlage (logic/solar_env.py) — keine Aktion (nicht steuerbar), eigene Erzeugung.
Beide senken die Kosten, wenn sie Netzbezug ersetzen, ohne dass BOPTEST selbst davon etwas
mitbekommt — siehe workbench.md für die genaue Kostenrechnung.

BOPTEST läuft als eigener Docker-Dienst (siehe workbench.md) und wird ausschließlich über
REST angesprochen — hier nur die Verbindungsparameter, keine Gebäudesimulation im
Python-Prozess selbst.
"""
import random

from logic.battery_env import BatteryEnv
from logic.boptest_gym_env import BoptestGymEnv
from logic.solar_env import SolarEnv

URL = 'http://127.0.0.1:8000'
TESTCASE = 'bestest_hydronic_heat_pump'

# Beobachtungen: Zonentemperatur + Komfortband, Außentemperatur, Solareinstrahlung,
# dynamischer Strompreis — alles, was ein vorausschauender Agent braucht, um günstigen Strom
# zu nutzen, ohne den Komfort zu verletzen. `time` normalisiert auf eine Wochenperiode, damit
# der Agent Wochentag/Uhrzeit-Muster erkennen kann. BatteryEnv/SolarEnv hängen zusätzlich
# Ladezustand bzw. aktuelle PV-Erzeugung an (siehe dort) — hier nur die BOPTEST-seitigen.
OBSERVATIONS = {
    'time': (0, 7 * 24 * 3600),
    'reaTZon_y': (280., 310.),                    # Zonentemperatur (K)
    'TDryBul': (265., 303.),                       # Außentemperatur (K)
    'HDirNor': (0., 862.),                          # Direkte Solareinstrahlung (W/m²)
    'InternalGainsRad[1]': (0., 219.),                # Interne Wärmelasten (W)
    'PriceElectricPowerDynamic': (-0.4, 0.4),           # Dynamischer Strompreis
    'LowerSetp[1]': (280., 310.),                        # Komfortband: untere Grenze
    'UpperSetp[1]': (280., 310.),                         # Komfortband: obere Grenze
}
# Wärmepumpen-Modulationssignal (0 = aus, 1 = volle Leistung) — die einzige Aktion, die
# tatsächlich an BOPTEST geht. BatteryEnv hängt eine zweite, rein lokale Aktion an
# (Batterieleistung in kW, siehe logic/battery_env.py); SolarEnv fügt keine Aktion hinzu
# (Solarerzeugung ist nicht steuerbar) — hier nur die BOPTEST-seitige Aktion.
ACTIONS = ['oveHeaPumY_u']

# Testperiode (analog zum train/test-Split der bisherigen CityLearn-Version): der Rest des
# Jahres zum Trainieren, eine feste, nie im Training gesehene Periode zum Testen. Die ersten
# drei Februartage sind eine im BOPTEST-Ökosystem gängige Testperiode für diesen Testfall
# (siehe BOPTEST-Gym-Beispiele), mit drei Tagen Einschwingzeit davor.
TEST_START = 31 * 24 * 3600         # 1. Februar (Sekunden seit Jahresbeginn)
TEST_LENGTH = 3 * 24 * 3600         # 3 Tage
TEST_WARMUP = 3 * 24 * 3600         # 3 Tage Einschwingzeit direkt davor
TRAIN_EXCLUDE = [(TEST_START - TEST_WARMUP, TEST_START + TEST_LENGTH)]

PREDICTIVE_PERIOD = 24 * 3600   # 24h Preis-/Wetter-Vorschau in der Beobachtung
STEP_PERIOD = 3600              # stündliche Regelschritte
TRAIN_EPISODE_LENGTH = 7 * 24 * 3600   # eine Woche pro Trainingsepisode
TRAIN_WARMUP = 24 * 3600               # 1 Tag Einschwingzeit pro Trainingsepisode


def make_env(split='train', seed=0, reward_kwargs=None, battery_kwargs=None, solar_kwargs=None,
            battery=True, solar=True, url=URL, testcase=TESTCASE, scenario=None):
    """BOPTEST-Gymnasium-Umgebung, standardmäßig mit Batterie + PV-Anlage (siehe
    logic/battery_env.py, logic/solar_env.py).

    split='train': zufälliger Startzeitpunkt übers Jahr, Testperiode ausgeschlossen.
    split='test' : feste, nie im Training gesehene Periode (siehe TEST_START/-LENGTH).
    reward_kwargs: an BatteryEnv weitergereicht, z. B. {'w_comfort': 3.0}.
    battery_kwargs: an BatteryEnv weitergereicht, z. B. {'capacity_kwh': 13.5, 'max_power_kw': 5.0}.
    solar_kwargs: an SolarEnv weitergereicht, z. B. {'panel_area_m2': 30.0}.
    battery/solar: False = die jeweilige Schicht weglassen (solar=True ohne battery=True
                  funktioniert, senkt dann aber die Kosten nur, wenn battery=True zusätzlich
                  den Netzbezug in der Beobachtung/im info-Dict bereitstellt).
    scenario: BOPTEST-Preisszenario, Standard 'dynamic' (stündlich variierender Day-Ahead-Preis).
    """
    random.seed(seed)
    reward_kwargs = reward_kwargs or {}
    battery_kwargs = battery_kwargs or {}
    solar_kwargs = solar_kwargs or {}
    scenario = scenario or {'electricity_price': 'dynamic'}

    common = dict(url=url, testcase=testcase, actions=ACTIONS, observations=OBSERVATIONS,
                 predictive_period=PREDICTIVE_PERIOD, step_period=STEP_PERIOD, scenario=scenario)

    if split == 'test':
        env = BoptestGymEnv(random_start_time=False, start_time=TEST_START,
                            max_episode_length=TEST_LENGTH, warmup_period=TEST_WARMUP, **common)
    else:
        env = BoptestGymEnv(random_start_time=True, excluding_periods=TRAIN_EXCLUDE,
                            max_episode_length=TRAIN_EPISODE_LENGTH, warmup_period=TRAIN_WARMUP, **common)

    if battery:
        env = BatteryEnv(env, **reward_kwargs, **battery_kwargs)
    if solar:
        env = SolarEnv(env, **solar_kwargs)
    return env
