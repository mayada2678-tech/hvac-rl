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
# Zwischenspeicher-Version: Zeilen älterer Versionen (z. B. ohne die vollständigen BOPTEST-KPIs)
# werden beim nächsten Vergleich neu simuliert statt mit Lücken angezeigt.
CACHE_VERSION = 2

# Alle Kennzahlen, die im Reiter "Vergleich" frei auf die Achsen gelegt werden können:
# Schlüssel -> (Beschriftung, besser ist 'low'/'high'/None). BOPTEST-KPIs laut
# https://ibpsa.github.io/project1-boptest/docs-testcases/ (Werte je m² Wohnfläche bzw. je Zone).
# Kennzahlen ohne Werte (z. B. Gasverbrauch bei einer Wärmepumpe) blendet die Oberfläche aus.
METRICS = {
    'cost_eur':         ('Stromkosten inkl. Batterie/PV (€)', 'low'),
    'savings_pct':      ('Ersparnis ggü. Referenz (%)', 'high'),
    'tdis_kh':          ('Komfortverletzung (K·h)', 'low'),
    'grid_kwh':         ('Netzbezug (kWh)', 'low'),
    'cost_tot_boptest': ('Kosten lt. BOPTEST, ohne Batterie/PV (€/m²)', 'low'),
    'ener_tot':         ('Energieverbrauch lt. BOPTEST (kWh/m²)', 'low'),
    'emis_tot':         ('CO₂-Emissionen lt. BOPTEST (kg/m²)', 'low'),
    'pele_tot':         ('Elektrische Spitzenlast (kW/m²)', 'low'),
    'pgas_tot':         ('Gas-Spitzenlast (kW/m²)', 'low'),
    'pdih_tot':         ('Fernwärme-Spitzenlast (kW/m²)', 'low'),
    'idis_tot':         ('Luftqualitäts-Defizit (ppm·h)', 'low'),
    'time_rat':         ('Rechenzeit je Schritt (relativ)', 'low'),
    'w':                ('Komfortgewicht w', None),
}
BOPTEST_KPIS = ['ener_tot', 'emis_tot', 'pele_tot', 'pgas_tot', 'pdih_tot', 'idis_tot', 'time_rat']


def run_episode(policy) -> dict:
    """Eine Testepisode. policy: 'rbc' oder Pfad zu einem Modell (ohne .zip). Ergebnis:
    BOPTESTs KPIs plus cost_eur (Kosten inkl. Batterie/PV) und grid_kwh (Netzbezug)."""
    is_rbc = isinstance(policy, str) and policy == 'rbc'
    env = make_env('test', normalize=is_rbc or obs_normalized(Path(policy).with_suffix('.zip')))
    try:
        if is_rbc:
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


def obs_normalized(model_zip: Path) -> bool:
    """Wurde das Modell mit normierten Beobachtungen trainiert? Steht in den Metadaten
    (models/<name>.json, auch für <name>_final.zip); fehlt das Feld, ist es ein älteres
    Modell, das die Rohwerte gesehen hat."""
    model_zip = Path(model_zip)
    stem = model_zip.stem.removesuffix('_final')
    return bool(model_meta(model_zip.with_name(stem + '.zip')).get('obs_normalized', False))


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
            cache = {r['name']: r for r in old.to_dict('records')
                     if not isinstance(r.get('error'), str) and r.get('cache_version') == CACHE_VERSION}
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
                         'cost_tot_boptest': r.get('cost_tot'), 'mtime': mtime, 'error': None,
                         'cache_version': CACHE_VERSION, **{k: r.get(k) for k in BOPTEST_KPIS}})
        except Exception as e:   # z. B. altes Modell mit anderem Beobachtungsraum
            rows.append({'name': name, 'algo': algo, 'w': w, 'mtime': mtime, 'error': str(e)[:200]})
    if progress:
        progress(len(todo), len(todo), '')

    df = pd.DataFrame(rows).reindex(columns=COMPARE_COLUMNS + BOPTEST_KPIS + ['cache_version'])
    df['savings_pct'] = savings_vs(df, RBC)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def savings_vs(df: pd.DataFrame, reference: str = RBC) -> pd.Series:
    """Ersparnis jeder Zeile gegenüber den Stromkosten der Referenz (RBC oder ein Modell), in %."""
    ref = df.loc[df.name == reference, 'cost_eur']
    if not len(ref) or pd.isna(ref.iloc[0]) or ref.iloc[0] <= 0:
        return pd.Series(np.nan, index=df.index)
    return (1 - df['cost_eur'] / ref.iloc[0]) * 100


def display_name(row) -> str:
    """'SAC w=0.1' für Studien-Modelle, sonst der Modellname bzw. 'RBC'."""
    if row['name'] == RBC:
        return 'RBC'
    if pd.notna(row.get('w')):
        return f"{row['algo']} w={row['w']:g}"
    return row['name']


def summary_sentences(df: pd.DataFrame, reference: str = RBC, names=None) -> list[str]:
    """Kernaussage in Worten gegenüber einer frei wählbaren Referenz (RBC oder ein Modell), je
    Modell eine Zeile, sortiert nach Algorithmus und absteigendem w — z. B. 'SAC w=0.1: 18.3 %
    günstiger als der RBC, aber 12.4 K·h Komfortverletzung (+11.5 K·h ggü. RBC).'
    names: nur diese Zeilen (Modellauswahl der Oberfläche); die Referenz zählt immer mit."""
    ok = df[df['error'].isna() & df['cost_eur'].notna()].copy()
    ref = ok[ok.name == reference]
    if ref.empty:
        return []
    ok['savings_pct'] = savings_vs(ok, reference)
    ref_row = ref.iloc[0]
    ref_name = display_name(ref_row)
    ref_article = 'der RBC' if reference == RBC else ref_name
    ref_tdis = float(ref_row.tdis_kh)
    lines = [f'Referenz {ref_name}: {ref_row.cost_eur:.2f} € Stromkosten, '
             f'{ref_tdis:.1f} K·h Komfortverletzung im Testzeitraum.']
    others = ok[ok.name != reference]
    if names is not None:
        others = others[others.name.isin(list(names))]
    others = others.sort_values(['algo', 'w'], ascending=[True, False], na_position='last')
    for _, r in others.iterrows():
        pct = r.savings_pct
        if pd.isna(pct):
            cost_txt = f'{r.cost_eur:.2f} €'
        elif abs(pct) < 1:
            cost_txt = f'kaum Kostenunterschied zu {ref_article} ({pct:+.1f} %)'
        elif pct > 0:
            cost_txt = f'{pct:.1f} % günstiger als {ref_article}'
        else:
            cost_txt = f'{-pct:.1f} % teurer als {ref_article}'
        comfort = r.tdis_kh - ref_tdis
        cheaper = pd.notna(pct) and pct >= 1
        # Bindewort nach Richtung, Komma nur vor Gegensatz ("aber"/"dafür"):
        # günstiger + schlechterer Komfort = "aber", teurer + schlechterer Komfort = "und",
        # günstiger + gleich guter Komfort = "bei", teurer + gleich guter Komfort = "dafür".
        if comfort > 0.05:
            joint = ', aber' if cheaper else ' und'
            comfort_txt = (f'{joint} {r.tdis_kh:.1f} K·h Komfortverletzung '
                           f'({comfort:+.1f} K·h ggü. {"RBC" if reference == RBC else ref_name})')
        else:
            joint = ' bei' if cheaper else ', dafür'
            comfort_txt = f'{joint} {r.tdis_kh:.1f} K·h Komfortverletzung (nicht mehr als {ref_article})'
        lines.append(f'{display_name(r)}: {cost_txt}{comfort_txt}.')
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
