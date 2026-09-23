"""Läuft eine Testepisode mit einer Strategie und zeichnet alles auf, was man zum Beobachten
braucht: Zonentemperatur, Komfortband, Wärmepumpen-Aktion, Batterie (Aktion + Ladezustand),
PV-Erzeugung, Außentemperatur, Strompreis, Belohnung — eine Zeile je Stunde.
"""
import numpy as np
import pandas as pd

from logic.baselines import build
from logic.envs import make_env

# Messgrößen, die direkt über BOPTESTs /results-Endpunkt abrufbar sind (siehe
# boptest-gym/examples/test_and_plot.py::plot_results für dasselbe Muster).
RESULT_POINTS = ['reaTZon_y', 'reaTSetHea_y', 'reaTSetCoo_y', 'oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y']


def policy_from(name_or_model, split='test'):
    """Gibt (env, act_fn) zurück. name: 'rbc' | ein geladenes SB3-Modell. `env` ist die
    batteriefähige Umgebung (logic.battery_env.BatteryEnv), die eine BOPTEST-Umgebung umhüllt
    — siehe logic/envs.py::make_env()."""
    env = make_env(split)
    if isinstance(name_or_model, str):
        agent = build(name_or_model, env)
        return env, lambda obs: agent.predict(obs)[0]
    return env, lambda obs: name_or_model.predict(obs, deterministic=True)[0]


def record(name_or_model, split='test') -> tuple[pd.DataFrame, dict]:
    """Eine Testepisode durchspielen. Ergebnis: (DataFrame mit einer Zeile je Stunde —
    Spalten t, hour, indoor, setpoint_heat, setpoint_cool, outdoor, heat_pump_action,
    battery_power, battery_soc, solar_power, grid_power, price, reward —, BOPTESTs offizielle
    KPIs für diese Episode)."""
    env, act = policy_from(name_or_model, split)
    base = env.unwrapped
    obs, info = env.reset()

    rewards, battery_power, battery_soc = [], [], []
    solar_power, grid_power, prices = [], [], []
    done, t = False, 0
    while not done:
        action = act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(float(reward))
        battery_power.append(float(info.get('battery_power_kw', 0.0)))
        battery_soc.append(float(info.get('battery_soc', np.nan)))
        solar_power.append(float(info.get('solar_power_kw', 0.0)))
        grid_power.append(float(info.get('grid_power_kw', np.nan)))
        prices.append(float(info.get('price', np.nan)))
        t += 1

    res = base.get_results(RESULT_POINTS, start_time=base.start_time + 1)
    series = pd.DataFrame(res)
    kpis = base.get_kpis()
    base.stop()

    n = len(rewards)
    # BOPTESTs /results-Zeitraster kann feiner als unsere Stundenschritte sein (interne
    # Lösungsschritte des FMU) — auf die Stunden reindizieren, zu denen der Agent tatsächlich
    # gehandelt hat.
    target_times = base.start_time + base.step_period * np.arange(1, n + 1)
    idx = np.searchsorted(series['time'].to_numpy(), target_times)
    idx = np.clip(idx, 0, len(series) - 1)
    aligned = series.iloc[idx].reset_index(drop=True)

    df = pd.DataFrame({
        't': np.arange(n),
        'hour': (target_times // 3600 % 24).astype(int),
        'indoor': aligned['reaTZon_y'] - 273.15,
        'setpoint_heat': aligned['reaTSetHea_y'] - 273.15,
        'setpoint_cool': aligned['reaTSetCoo_y'] - 273.15,
        'outdoor': aligned['weaSta_reaWeaTDryBul_y'] - 273.15,
        'heat_pump_action': aligned['oveHeaPumY_u'],
        'battery_power': battery_power,
        'battery_soc': battery_soc,
        'solar_power': solar_power,
        'grid_power': grid_power,
        'price': prices,
        'reward': rewards,
    })
    return df, kpis
