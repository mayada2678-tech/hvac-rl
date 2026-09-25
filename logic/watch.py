"""Läuft eine Testepisode mit einer Strategie und zeichnet alles auf, was man zum Beobachten
braucht: Zonentemperatur, Komfortband, Wärmepumpen-Aktion, Batterie (Aktion + Ladezustand),
PV-Erzeugung, Außentemperatur, Strompreis, Belohnung — eine Zeile je Stunde.

Nebenbei landet nach jedem Schritt der aktuelle Zustand in runs/live/anim_state.json — die
Live-Animation web/hvac-agent-animation.html liest die Datei über scripts/serve_animation.py.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from logic.baselines import build
from logic.envs import make_env
from logic.reward import REWARD_PARTS

# Messgrößen, die direkt über BOPTESTs /results-Endpunkt abrufbar sind (siehe
# boptest-gym/examples/test_and_plot.py::plot_results für dasselbe Muster).
RESULT_POINTS = ['reaTZon_y', 'reaTSetHea_y', 'reaTSetCoo_y', 'oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y']
ANIM_STATE_PATH = Path(__file__).resolve().parent.parent / 'runs' / 'live' / 'anim_state.json'


def _kelvin_to_c(v):
    return None if v is None else float(v) - 273.15


def write_anim_state(path: Path, state: dict):
    """Atomar schreiben (temporäre Datei + os.replace), damit der Browser nie eine halb
    geschriebene Datei liest. Unter Windows kann os.replace scheitern, solange der Server die
    Datei gerade offen hat — dann diesen Schritt auslassen, der nächste kommt gleich."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    # NaN ist kein gültiges JSON (der Browser würde die ganze Datei verwerfen) -> null.
    state = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in state.items()}
    tmp.write_text(json.dumps(state), encoding='utf-8')
    try:
        os.replace(tmp, path)
    except PermissionError:
        pass


def anim_state(info: dict, strategy: str, step: int, n_steps: int | None = None, done: bool = False,
               action=None, reward: float | None = None) -> dict:
    """Zustand eines Schritts für die Animation (web/hvac-agent-animation.html) aus dem
    info-Dict der Umgebung. action=None: aus BOPTESTs Antwort (oveHeaPumY_u) und der
    tatsächlichen Batterieleistung ableiten — so beim Training (logic/training.py)."""
    res = info.get('res', {})
    if action is None:
        hp = res.get('oveHeaPumY_u')
        action = [] if hp is None else [hp, info.get('battery_power_kw', 0.0)]
    return {
        'strategy': strategy, 'step': step, 'n_steps': n_steps, 'done': bool(done),
        'time_s': float(res.get('time', 0.0)),
        'price': float(info.get('price', np.nan)), 'battery_soc': float(info.get('battery_soc', np.nan)),
        'battery_power_kw': float(info.get('battery_power_kw', 0.0)),
        'solar_power_kw': float(info.get('solar_power_kw', 0.0)),
        'grid_power_kw': float(info.get('grid_power_kw', 0.0)),
        'heat_pump_power_kw': float(info.get('heat_pump_power_kw', 0.0)),
        'reaTZon_y': res.get('reaTZon_y'), 'indoor_c': _kelvin_to_c(res.get('reaTZon_y')),
        'setpoint_heat_c': _kelvin_to_c(res.get('reaTSetHea_y')),
        'setpoint_cool_c': _kelvin_to_c(res.get('reaTSetCoo_y')),
        'outdoor_c': _kelvin_to_c(res.get('weaSta_reaWeaTDryBul_y')),
        # Aktion [oveHeaPumY_u, battery_power_kw]; leer beim RBC (BOPTESTs eingebauter Regler
        # steuert die Wärmepumpe selbst).
        'action': [float(a) for a in action],
        'reward': None if reward is None else float(reward), 'updated': time.time(),
    }


def policy_from(name_or_model, split='test'):
    """Gibt (env, act_fn) zurück. name: 'rbc' | ein geladenes SB3-Modell. `env` ist die
    batteriefähige Umgebung (logic.battery_env.BatteryEnv), die eine BOPTEST-Umgebung umhüllt
    — siehe logic/envs.py::make_env()."""
    env = make_env(split)
    if isinstance(name_or_model, str):
        agent = build(name_or_model, env)
        return env, lambda obs: agent.predict(obs)[0]
    return env, lambda obs: name_or_model.predict(obs, deterministic=True)[0]


def record(name_or_model, split='test', anim_state_path: Path | None = ANIM_STATE_PATH) -> tuple[pd.DataFrame, dict]:
    """Eine Testepisode durchspielen. Ergebnis: (DataFrame mit einer Zeile je Stunde —
    Spalten t, hour, indoor, setpoint_heat, setpoint_cool, outdoor, heat_pump_action,
    battery_power, battery_soc, solar_power, grid_power, heat_pump_power, price, reward, reward_cost,
    reward_comfort, reward_battery, reward_terminal —, BOPTESTs offizielle
    KPIs für diese Episode). anim_state_path=None schaltet die Live-Datei ab."""
    env, act = policy_from(name_or_model, split)
    base = env.unwrapped
    obs, info = env.reset()

    rewards, battery_power, battery_soc = [], [], []
    solar_power, grid_power, prices, hp_power = [], [], [], []
    parts = {k: [] for k in REWARD_PARTS}
    strategy = name_or_model if isinstance(name_or_model, str) else type(name_or_model).__name__
    n_steps = int(getattr(base, 'max_episode_length', 0) // base.step_period) or None
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
        hp_power.append(float(info.get('heat_pump_power_kw', 0.0)))
        for k in REWARD_PARTS:
            parts[k].append(float(info.get(k, 0.0)))
        if anim_state_path is not None:
            write_anim_state(anim_state_path, anim_state(info, strategy, t + 1, n_steps, done,
                                                         action=np.atleast_1d(action), reward=reward))
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
        'heat_pump_power': hp_power,
        'price': prices,
        'reward': rewards,
        **parts,
    })
    return df, kpis
