"""Kennzahlen von Baselines und trainierten Modellen auf der Testperiode.

Zwei Sichten:
- BOPTESTs eigene, offizielle KPIs (`cost_tot`, `tdis_tot`, Emissionen, ...) — direkt mit
  anderen BOPTEST-Ergebnissen vergleichbar, kennen aber Batterie und PV nicht (siehe
  logic/battery_env.py).
- Die tatsächlichen Kosten dieses Projekts inkl. Batterie und PV (`cost_eur`, Summe von
  `info['step_cost']`) — Grundlage des Kosten-Komfort-Vergleichs (`compare()`), mit dem sich
  mehrere Komfortgewichte w gegeneinander und gegen den RBC stellen lassen:
  „Bei w=1 kaum Ersparnis, bei w=0.1 sparen wir X %, aber Y Kelvin-Stunden Komfortverletzung“.

CLI: python -m logic.evaluation   (RBC + alle Modelle in models/, schreibt results/kpi.csv
                                   und results/compare.csv)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3

from logic.baselines import build
from logic.envs import make_env

ALGOS = {'sac': SAC, 'td3': TD3, 'ppo': PPO}
RBC = 'Regel (RBC)'
COMPARE_COLUMNS = ['name', 'algo', 'w', 'cost_eur', 'grid_kwh', 'tdis_kh', 'cost_tot_boptest',
                   'savings_pct', 'mtime', 'error']


def run_episode(policy) -> dict:
    """Eine Testepisode. policy: 'rbc' oder Pfad zu einem Modell (ohne .zip). Ergebnis:
    BOPTESTs KPIs plus cost_eur (Kosten inkl. Batterie/PV) und grid_kwh (Netzbezug)."""
    env = make_env('test')
    try:
        if isinstance(policy, str) and policy == 'rbc':
            agent = build('rbc', env)
            act = lambda obs: agent.predict(obs)[0]  # noqa: E731
        else:
            path = Path(policy)
            model = ALGOS[path.name.split('_')[0]].load(path)
            act = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
        dt_hours = env.unwrapped.step_period / 3600.0
        obs, _ = env.reset()
        cost_eur = grid_kwh = 0.0
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
            cost_eur += float(info.get('step_cost', 0.0))
            grid_kwh += float(info.get('grid_power_kw', 0.0)) * dt_hours
        kpis = env.unwrapped.get_kpis()
    finally:
        env.close()
    return dict(kpis, cost_eur=cost_eur, grid_kwh=grid_kwh)


def run_baseline(name: str = 'rbc') -> dict:
    return run_episode(name)


def run_model(path: Path) -> dict:
    return run_episode(path)


def model_meta(model_zip: Path) -> dict:
    """Trainingsmetadaten neben dem Modell (logic/training.py schreibt models/<name>.json)."""
    meta_path = model_zip.with_suffix('.json')
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return {}


def comparable_models(models_dir: Path) -> list[Path]:
    """Die eigentlichen Modelle (bestes Zwischenmodell je Lauf) — ohne *_final-Endstände."""
    if not Path(models_dir).exists():
        return []
    return sorted(p for p in Path(models_dir).glob('*.zip') if not p.stem.endswith('_final'))


def compare(models_dir: Path = Path('models'), cache_path: Path = Path('results/compare.csv'),
            force: bool = False, progress=None) -> pd.DataFrame:
    """RBC + alle Modelle auf der Testperiode, eine Zeile je Modell: Kosten (€, inkl.
    Batterie/PV), Netzbezug (kWh), Komfortverletzung (K·h), Ersparnis ggü. RBC (%).

    Ergebnisse werden in cache_path zwischengespeichert und nur für neue oder seit der letzten
    Berechnung neu trainierte Modelle (Dateizeit) erneut simuliert — eine Testepisode kostet
    je Modell etwa eine halbe bis eine Minute. force=True rechnet alles neu.
    progress(i, n, name): optionaler Fortschritts-Rückruf (für die Oberfläche)."""
    cache = {}
    if not force and Path(cache_path).exists():
        try:
            old = pd.read_csv(cache_path)
            cache = {r['name']: r for r in old.to_dict('records') if not isinstance(r.get('error'), str)}
        except (OSError, ValueError, pd.errors.EmptyDataError):
            cache = {}

    todo = [(RBC, 'rbc', None, 0.0)] + [(p.stem, p.with_suffix(''), p, p.stat().st_mtime)
                                         for p in comparable_models(models_dir)]
    rows = []
    for i, (name, policy, zip_path, mtime) in enumerate(todo):
        meta = model_meta(zip_path) if zip_path else {}
        w = meta.get('reward', {}).get('w_comfort', np.nan) if zip_path else np.nan
        algo = name.split('_')[0].upper() if zip_path else 'RBC'
        cached = cache.get(name)
        if cached is not None and abs(float(cached.get('mtime', -1)) - mtime) < 1e-6:
            rows.append({**cached, 'w': w})
            continue
        if progress:
            progress(i, len(todo), name)
        try:
            r = run_episode(policy)
            rows.append({'name': name, 'algo': algo, 'w': w, 'cost_eur': r['cost_eur'],
                         'grid_kwh': r['grid_kwh'], 'tdis_kh': r.get('tdis_tot') or 0.0,
                         'cost_tot_boptest': r.get('cost_tot'), 'mtime': mtime, 'error': None})
        except Exception as e:   # z. B. altes Modell mit anderem Beobachtungsraum
            rows.append({'name': name, 'algo': algo, 'w': w, 'mtime': mtime, 'error': str(e)[:200]})
    if progress:
        progress(len(todo), len(todo), '')

    df = pd.DataFrame(rows).reindex(columns=COMPARE_COLUMNS)
    rbc_cost = df.loc[df.name == RBC, 'cost_eur']
    if len(rbc_cost) and pd.notna(rbc_cost.iloc[0]) and rbc_cost.iloc[0] > 0:
        df['savings_pct'] = (1 - df['cost_eur'] / rbc_cost.iloc[0]) * 100
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def summary_sentences(df: pd.DataFrame) -> list[str]:
    """Kernaussage des Kosten-Komfort-Vergleichs in Worten, je Modell eine Zeile, sortiert
    nach Algorithmus und absteigendem w — z. B. 'SAC, w=0.1: 18.3 % günstiger als der RBC,
    aber 12.4 K·h Komfortverletzung (RBC: 0.9 K·h).'"""
    ok = df[df['error'].isna() & df['cost_eur'].notna()]
    rbc = ok[ok.name == RBC]
    if rbc.empty:
        return []
    rbc_tdis = float(rbc.tdis_kh.iloc[0])
    lines = [f'RBC (Referenz): {rbc.cost_eur.iloc[0]:.2f} € Stromkosten, '
             f'{rbc_tdis:.1f} K·h Komfortverletzung im Testzeitraum.']
    models = ok[ok.name != RBC].sort_values(['algo', 'w'], ascending=[True, False], na_position='last')
    for _, r in models.iterrows():
        w_txt = f'w={r.w:g}' if pd.notna(r.w) else r['name']
        pct = r.savings_pct
        if pd.isna(pct):
            cost_txt = f'{r.cost_eur:.2f} €'
        elif abs(pct) < 1:
            cost_txt = f'kaum Ersparnis ({pct:+.1f} %)'
        elif pct > 0:
            cost_txt = f'{pct:.1f} % günstiger als der RBC'
        else:
            cost_txt = f'{-pct:.1f} % teurer als der RBC'
        comfort = r.tdis_kh - rbc_tdis
        worse_comfort = comfort > 0.05
        cheaper = pd.notna(pct) and pct >= 1
        # Bindewort nach Richtung: günstiger/schlechterer Komfort = "aber", teurer/schlechterer
        # Komfort = "und", teurer/gleich guter Komfort = "dafür", sonst "bei".
        # Komma nur vor "aber"/"dafür" (Gegensatz), nicht vor "und"/"bei".
        if worse_comfort:
            joint = ', aber' if cheaper else ' und'
            comfort_txt = f'{joint} {r.tdis_kh:.1f} K·h Komfortverletzung ({comfort:+.1f} K·h ggü. RBC)'
        else:
            joint = ' bei' if cheaper else ', dafür'
            comfort_txt = f'{joint} {r.tdis_kh:.1f} K·h Komfortverletzung (nicht mehr als der RBC)'
        lines.append(f'{r.algo}, {w_txt}: {cost_txt}{comfort_txt}.')
    return lines


def evaluate_all(models_dir: Path = Path('models')) -> pd.DataFrame:
    """RBC-Baseline plus alle *.zip-Modelle in models_dir, jeweils mit BOPTESTs KPIs."""
    rows = {RBC: run_baseline('rbc')}
    for p in sorted(Path(models_dir).glob('*.zip')):
        rows[p.stem] = run_model(p.with_suffix(''))
    return pd.DataFrame(rows).T


if __name__ == '__main__':
    table = evaluate_all()
    print(table.round(4).to_string())
    Path('results').mkdir(exist_ok=True)
    table.to_csv('results/kpi.csv')
    print('\n'.join(summary_sentences(compare())))
