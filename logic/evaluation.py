"""Kennzahlen (KPIs) von Baselines und trainierten Modellen auf der Testperiode — BOPTESTs
eigene, offizielle KPI-Berechnung (Kosten, Komfort-Defizit, Emissionen, ...), nicht
selbstgebaute Kennzahlen. Damit sind Werte direkt mit anderen BOPTEST-Ergebnissen vergleichbar.

CLI: python -m logic.evaluation   (wertet die RBC-Baseline und alle Modelle in models/ aus,
                                   schreibt results/kpi.csv)
"""
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO, SAC, TD3

from logic.baselines import build
from logic.envs import make_env

ALGOS = {'sac': SAC, 'td3': TD3, 'ppo': PPO}


def run_baseline(name: str = 'rbc') -> dict:
    env = make_env('test')
    agent = build(name, env)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    kpis = env.unwrapped.get_kpis()
    env.unwrapped.stop()
    return kpis


def run_model(path: Path) -> dict:
    env = make_env('test')
    model = ALGOS[path.name.split('_')[0]].load(path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    kpis = env.unwrapped.get_kpis()
    env.unwrapped.stop()
    return kpis


def evaluate_all(models_dir: Path = Path('models')) -> pd.DataFrame:
    """RBC-Baseline plus alle *.zip-Modelle in models_dir, jeweils mit BOPTESTs KPIs."""
    rows = {'Regel (RBC)': run_baseline('rbc')}
    for p in sorted(Path(models_dir).glob('*.zip')):
        rows[p.stem] = run_model(p.with_suffix(''))
    return pd.DataFrame(rows).T


if __name__ == '__main__':
    table = evaluate_all()
    print(table.round(4).to_string())
    Path('results').mkdir(exist_ok=True)
    table.to_csv('results/kpi.csv')
