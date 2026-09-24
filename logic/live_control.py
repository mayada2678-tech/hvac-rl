"""Gemeinsame Hilfsfunktionen für die Live-Trainingssteuerung.

Wird sowohl von logic/training.py (läuft als eigener Prozess) als auch von
gui/agent_page.py (Streamlit-Oberfläche) verwendet. Die beiden reden ausschließlich über
Dateien miteinander (Steuerdatei, Statusdatei, CSV-Lernkurve) — kein Multiprocessing,
kein gemeinsamer Speicher, funktioniert deshalb unter Windows problemlos und
das Training läuft unabhängig vom Streamlit-Prozess weiter.
"""
import json
from pathlib import Path

import yaml

from logic.reward import REWARD_DEFAULTS

ALGOS = ['SAC', 'PPO', 'TD3']

# BOPTEST-Worker (scripts/start_boptest.ps1 -Workers, Standard 6). Jedes Training belegt zwei
# (Training + Auswertung) -> so viele Trainings laufen gleichzeitig, weitere warten in der
# Oberfläche, bis eines fertig ist.
BOPTEST_WORKERS = 6
MAX_PARALLEL_TRAININGS = BOPTEST_WORKERS // 2
ALGO_CONFIGS = {'SAC': 'configs/sac.yaml', 'PPO': 'configs/ppo.yaml', 'TD3': 'configs/td3.yaml'}

# "Beste theoretische Empfehlung" je Algorithmus: Netzwerkgröße und Hyperparameter, wie sie
# die Originalarbeiten bzw. Stable-Baselines3 selbst als Standard verwenden — SAC (Haarnoja
# et al. 2018 / SB3-Default net_arch=[256,256]), TD3 (Fujimoto et al. 2018 / SB3-Default
# net_arch=[400,300]), PPO (Schulman et al. 2017 / SB3-Default net_arch=[64,64]). Das ist der
# Vorbelegungswert in der Oberfläche; total_timesteps/eval_freq/buffer_size sind bewusst
# klein gehalten (BOPTEST-Schritte sind REST-Aufrufe an den Docker-Dienst, keine reinen
# In-Memory-Berechnungen wie bei der früheren CityLearn-Version — jeder Schritt kostet
# spürbar Zeit). Jedes Feld lässt sich in der Oberfläche ändern.
RECOMMENDED = {
    'SAC': {
        'net_arch': [256, 256], 'total_timesteps': 20_000, 'eval_freq': 2_000,
        'params': {'learning_rate': 3e-4, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005,
                  'buffer_size': 20_000, 'train_freq': 1, 'gradient_steps': 1, 'ent_coef': 'auto'},
    },
    'PPO': {
        'net_arch': [64, 64], 'total_timesteps': 20_000, 'eval_freq': 2_016,
        'params': {'learning_rate': 3e-4, 'n_steps': 504, 'batch_size': 126, 'n_epochs': 10,
                  'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.0},
    },
    'TD3': {
        'net_arch': [400, 300], 'total_timesteps': 20_000, 'eval_freq': 2_000,
        'params': {'learning_rate': 1e-3, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005,
                  'buffer_size': 20_000, 'train_freq': 1, 'gradient_steps': 1,
                  'policy_delay': 2, 'target_policy_noise': 0.2, 'target_noise_clip': 0.5},
    },
}
# Belohnungsparameter, siehe logic/reward.py (dort Bedeutung und Quelle jedes Werts).
DEFAULT_REWARD = dict(REWARD_DEFAULTS)


def load_yaml_preset(algo: str, root: Path) -> dict:
    """Das projekteigene, per Hand getunte configs/<algo>.yaml als Alternative zur Literatur-Empfehlung."""
    cfg = yaml.safe_load((Path(root) / ALGO_CONFIGS[algo]).read_text())
    base = RECOMMENDED[algo]
    return {'net_arch': base['net_arch'], 'total_timesteps': cfg['total_timesteps'],
           'eval_freq': cfg.get('eval_freq', base['eval_freq']),
           'params': {**base['params'], **cfg.get('params', {})}}


def run_tag(algo: str, seed: int, variant: str | None = None) -> str:
    """Name eines Laufs = Name des Modells in models/, z. B. 'sac_seed0' oder mit Variante
    (Komfortgewicht-Studie) 'sac_seed0_w0_3'."""
    tag = f'{algo.lower()}_seed{seed}'
    return f'{tag}_{variant}' if variant else tag


def variant_for_w(w: float) -> str:
    """'w0_3' für w=0.3 — bewusst ohne Punkt: Stable-Baselines3 hält sonst '.3' für die
    Dateiendung und speichert das Modell ohne '.zip' (dann taucht es nirgends mehr auf)."""
    return f'w{w:g}'.replace('.', '_')


def paths_for(algo: str, seed: int, root: Path, variant: str | None = None) -> dict:
    """Alle Dateien eines Laufs: erzeugte Trainingsconfig, Lernkurve (CSV), Steuerdatei,
    Statusdatei, Konsolen-Log."""
    run_dir = Path(root) / 'runs' / 'live'
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = run_tag(algo, seed, variant)
    return {
        'run_config': run_dir / f'{tag}.run.json',
        'csv': run_dir / f'{tag}.csv',
        'control': run_dir / f'{tag}.control.json',
        'status': run_dir / f'{tag}.status.json',
        'log': run_dir / f'{tag}.log',
    }


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_json(path, **kwargs):
    Path(path).write_text(json.dumps(kwargs))


def read_control(path) -> dict:
    return read_json(path, {'pause': False, 'stop': False})


def write_control(path, **kwargs):
    write_json(path, **kwargs)


def read_status(path) -> dict:
    return read_json(path, {'phase': 'unbekannt'})
