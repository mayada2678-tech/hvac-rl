"""Trainingslogik — sowohl per Kommandozeile als auch aus der Oberfläche heraus nutzbar,
beides über dieselbe train()-Funktion (keine zwei parallelen Trainingsskripte mehr).

Kommandozeile (einfaches Training, wie ein gewöhnliches SB3-Skript):
    python -m logic.training --algo SAC --config configs/sac.yaml --seed 0

Aus der Oberfläche (gui/agent_page.py) heraus, als eigener Hintergrundprozess, mit
Live-Lernkurve und Fernsteuerung (Pause/Stopp) über drei zusätzliche Dateipfade:
    python -m logic.training --algo SAC --config runs/live/sac_seed0.run.json --seed 0 \
        --csv runs/live/sac_seed0.csv --control runs/live/sac_seed0.control.json \
        --status runs/live/sac_seed0.status.json

--config akzeptiert sowohl die handgepflegten configs/{sac,ppo,td3}.yaml als auch die von
der Oberfläche erzeugte *.run.json (beides gültiges YAML — siehe logic/live_control.py).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from logic.envs import make_env
from logic.live_control import anim_path_for, episodes_path_for, read_control, write_json
from logic.reward import REWARD_PARTS
from logic.watch import anim_state, write_anim_state

ALGOS = {'SAC': SAC, 'TD3': TD3, 'PPO': PPO}


class StopRequested(Exception):
    """Stopp kam mitten in einer Auswertungs-Episode — bricht sie sofort ab, statt die
    ganze Testperiode (72 Schritte, ~30-60 s) noch zu Ende zu rechnen."""


class LiveCallback(BaseCallback):
    """Wie EvalCallback (periodische Testepisode, bestes Modell sichern), zusätzlich:
    reagiert auf eine Steuerdatei (Pause/Stopp) und schreibt Lernkurve/Status in Dateien,
    damit eine separate Oberfläche den Fortschritt verfolgen kann. Jeder der drei Pfade ist
    einzeln optional — ohne control_path wird nie pausiert/gestoppt, ohne csv_path/status_path
    wird einfach nicht geschrieben.

    Die Steuerdatei wird höchstens alle 0.5s neu gelesen, damit das ständige Nachfragen
    das Training nicht durch Festplattenzugriffe pro Schritt ausbremst.
    """

    def __init__(self, eval_env, csv_path=None, control_path=None, status_path=None, eval_freq=5000,
                 n_eval_episodes=1, best_model_dir=None, status_extra=None, verbose=0):
        """status_extra: wird in jede Statuszeile mitgeschrieben (Prozess-ID, BOPTEST-Test-IDs),
        damit die Oberfläche den Prozess nach einem Neustart wiederfindet bzw. aufräumen kann."""
        super().__init__(verbose)
        self.status_extra = status_extra or {}
        self.eval_env = eval_env
        self.csv_path = Path(csv_path) if csv_path else None
        self.control_path = Path(control_path) if control_path else None
        self.status_path = Path(status_path) if status_path else None
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = n_eval_episodes
        self.best_model_dir = Path(best_model_dir) if best_model_dir else None
        self.best = -np.inf
        self._ctrl = {'pause': False, 'stop': False}
        self._last_poll = 0.0
        self._eval_parts = dict.fromkeys(REWARD_PARTS, 0.0)
        # Animation im Training-Reiter (gui/agent_view.py): aktueller Zustand, höchstens alle
        # 0,5 s geschrieben — beim Training aus der Trainingsumgebung, während einer Auswertung
        # aus der Testepisode.
        self.anim_path = anim_path_for(self.status_path) if self.status_path else None
        self._last_anim = 0.0
        # Belohnung jeder Trainingsepisode (vom Monitor-Wrapper), für die blasse Linie in der
        # Lernkurve — erscheint viel früher als die erste Auswertung.
        self.episodes_path = episodes_path_for(self.csv_path) if self.csv_path else None

    def _write_anim(self, info, phase, step, done, reward, force=False):
        now = time.time()
        if self.anim_path is None or not info or (not force and now - self._last_anim < 0.5):
            return
        self._last_anim = now
        write_anim_state(self.anim_path, anim_state(info, phase, step, done=done, reward=reward))

    def _poll_control(self):
        if self.control_path is None:
            return self._ctrl
        now = time.time()
        if now - self._last_poll > 0.5:
            self._last_poll = now
            self._ctrl = read_control(self.control_path)
        return self._ctrl

    def _write_status(self, **kwargs):
        if self.status_path is not None:
            write_json(self.status_path, **self.status_extra, **kwargs)

    def _check_stop_during_eval(self, _locals, _globals):
        # evaluate_policy ruft das je Schritt mit seinen lokalen Variablen auf — `info` trägt die
        # Einzelanteile der Belohnung aus logic/reward.py::RewardWrapper.
        info = _locals.get('info') or {}
        for k in REWARD_PARTS:
            self._eval_parts[k] += float(info.get(k, 0.0))
        i = _locals.get('i', 0)
        self._write_anim(info, 'Auswertung', int(_locals['current_lengths'][i]),
                         bool(_locals.get('done')), _locals.get('reward'), force=True)
        if self._poll_control().get('stop'):
            raise StopRequested

    def _on_step(self) -> bool:
        infos, dones = self.locals.get('infos') or [{}], self.locals.get('dones')
        done = bool(dones[0]) if dones is not None else False
        rewards = self.locals.get('rewards')
        self._write_anim(infos[0], 'Training', self.num_timesteps, done,
                         None if rewards is None else float(rewards[0]))
        episode = infos[0].get('episode')   # nur am Episodenende gesetzt (Monitor)
        if episode and self.episodes_path is not None:
            new = not self.episodes_path.exists()
            with open(self.episodes_path, 'a') as f:
                if new:
                    f.write('timesteps,reward,length\n')
                f.write(f"{self.num_timesteps},{float(episode['r'])},{int(episode['l'])}\n")
        ctrl = self._poll_control()
        if ctrl.get('stop'):
            self._write_status(phase='gestoppt', timesteps=self.num_timesteps)
            return False
        while ctrl.get('pause'):
            self._write_status(phase='pausiert', timesteps=self.num_timesteps)
            time.sleep(0.5)
            self._last_poll = time.time()
            self._ctrl = ctrl = read_control(self.control_path)
            if ctrl.get('stop'):
                self._write_status(phase='gestoppt', timesteps=self.num_timesteps)
                return False

        if self.num_timesteps % self.eval_freq == 0:
            self._write_status(phase='läuft', timesteps=self.num_timesteps, evaluating=True)
            self._eval_parts = dict.fromkeys(REWARD_PARTS, 0.0)
            try:
                mean_r, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                                            deterministic=True, callback=self._check_stop_during_eval)
            except StopRequested:
                self._write_status(phase='gestoppt', timesteps=self.num_timesteps)
                return False
            if self.csv_path is not None:
                new = not self.csv_path.exists()
                # Anteile wie mean_r als Mittel je Testepisode (zusammen = mean_r).
                parts = [self._eval_parts[k] / self.n_eval_episodes for k in REWARD_PARTS]
                with open(self.csv_path, 'a') as f:
                    if new:
                        f.write(','.join(['timesteps', 'reward', *REWARD_PARTS]) + '\n')
                    f.write(','.join(str(v) for v in [self.num_timesteps, mean_r, *parts]) + '\n')
            if self.best_model_dir is not None and mean_r > self.best:
                self.best = mean_r
                self.best_model_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_dir / 'best_model'))
            self._write_status(phase='läuft', timesteps=self.num_timesteps, reward=float(mean_r))
        else:
            self._write_status(phase='läuft', timesteps=self.num_timesteps)
        return True


def action_stats(model, env):
    """Eine Testepisode: Minimum/Maximum/Standardabweichung je Aktion (Wärmepumpe, Batterie),
    Endstand des Batterie-Ladezustands, batteriebewusste Gesamtkosten (siehe
    logic/battery_env.py) sowie BOPTESTs eigene KPIs (Kosten ohne Batterie-Kenntnis,
    Komfort-Defizit, ...) zum Vergleich."""
    obs, _ = env.reset()
    actions, info, done = [], {}, False
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    a = np.array(actions)
    print(f'\nWärmepumpen-Modulation: min {a[:, 0].min():.3f}  max {a[:, 0].max():.3f}  std {a[:, 0].std():.3f}')
    if a.shape[1] > 1:
        print(f'Batterieleistung (kW): min {a[:, 1].min():.3f}  max {a[:, 1].max():.3f}  std {a[:, 1].std():.3f}')
        print(f'Batterie-Ladezustand am Ende: {info.get("battery_soc", float("nan")):.2f}, '
             f'batteriebewusste Gesamtkosten: {getattr(env, "cum_cost", float("nan")):.4f}')
    kpis = env.unwrapped.get_kpis()
    print('KPIs (BOPTEST, ohne Batterie-Kenntnis):', {k: round(v, 4) for k, v in kpis.items() if v is not None})


def train(algo, config_path, seed=0, total_timesteps_override=None,
         csv_path=None, control_path=None, status_path=None, verbose=True):
    """Trainiert ein Modell und sichert Endstand + bestes Zwischenmodell nach models/.

    Ohne csv_path/control_path/status_path: normales Training wie von der Kommandozeile
    (druckt am Ende eine Aktionsstatistik). Mit mindestens einem der drei: Live-Lernkurve
    und/oder Fernsteuerung für eine Oberfläche, siehe LiveCallback.
    """
    cfg = yaml.safe_load(open(config_path))
    reward = cfg.get('reward', {})
    total_timesteps = total_timesteps_override or cfg['total_timesteps']

    # Prozess-ID sofort melden (noch vor dem langsamen Umgebungsaufbau), damit die Oberfläche
    # den Lauf von Anfang an zuordnen und notfalls beenden kann.
    status_extra = {'pid': os.getpid()}
    if status_path:
        write_json(status_path, **status_extra, phase='startet', timesteps=0)

    env = eval_env = None
    try:
        env = Monitor(make_env('train', seed=seed, reward_kwargs=reward))
        eval_env = Monitor(make_env('test', seed=seed, reward_kwargs=reward))
        status_extra['testids'] = [env.unwrapped.testid, eval_env.unwrapped.testid]
        if status_path:
            write_json(status_path, **status_extra, phase='startet', timesteps=0)
        return _train(algo, cfg, seed, total_timesteps, env, eval_env, reward,
                     csv_path, control_path, status_path, status_extra, verbose)
    finally:
        # BOPTEST-Tests immer freigeben — sonst bleiben die Worker belegt, bis BOPTEST sie
        # irgendwann selbst verwirft, und spätere Läufe scheitern mit KeyError: 'payload'.
        for e in (env, eval_env):
            if e is not None:
                e.close()


def _train(algo, cfg, seed, total_timesteps, env, eval_env, reward,
          csv_path, control_path, status_path, status_extra, verbose):

    # 'name' setzt die Oberfläche bei Varianten (z. B. sac_seed0_w0_3), sonst Standardname.
    name = cfg.get('name') or f'{algo.lower()}_seed{seed}'
    ckpt_dir = Path('models') / f'{name}_ckpt'
    net_arch = cfg.get('net_arch')
    policy_kwargs = dict(net_arch=net_arch) if net_arch else None

    live = bool(csv_path or control_path or status_path)
    model = ALGOS[algo]('MlpPolicy', env, seed=seed, verbose=(0 if live else 1),
                        tensorboard_log='runs', policy_kwargs=policy_kwargs, **cfg['params'])

    if live:
        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            for old in (Path(csv_path), episodes_path_for(csv_path)):
                if old.exists():
                    old.unlink()
        # Absichtlich KEIN Reset von control_path hier: Modellaufbau oben kann je nach
        # Systemlast spürbar dauern, und ein Reset an dieser Stelle würde einen Stopp/Pause-
        # Klick, der genau in diesem Fenster kam, stillschweigend überschreiben (realer Bug,
        # per Test reproduziert). Die Steuerdatei anzulegen ist Sache der aufrufenden Seite
        # (gui/agent_view.py schreibt sie, bevor dieser Prozess überhaupt gestartet wird);
        # fehlt sie doch einmal, liefert read_control() ohnehin sichere Standardwerte.
        callback = LiveCallback(eval_env, csv_path, control_path, status_path,
                                eval_freq=cfg.get('eval_freq', 5000), best_model_dir=ckpt_dir,
                                status_extra=status_extra)
    else:
        callback = EvalCallback(eval_env, eval_freq=cfg.get('eval_freq', 5000), n_eval_episodes=1,
                                deterministic=True, verbose=0, best_model_save_path=str(ckpt_dir))

    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=name)

    # Endstand sichern; falls es ein besseres Zwischenmodell laut Eval gab, wird das zum
    # eigentlichen Modell für logic/evaluation.py bzw. die "Agent beobachten"-Ansicht.
    out = Path('models') / f'{name}_final'
    model.save(out)
    best_path = ckpt_dir / 'best_model.zip'
    if best_path.exists():
        best = ALGOS[algo].load(best_path.with_suffix(''))
        best.save(Path('models') / name)
        model = best
    else:
        model.save(Path('models') / name)
    # Belohnungsparameter neben dem Modell ablegen — damit logic/evaluation.py weiß, mit
    # welchem Komfortgewicht dieses Modell trainiert wurde (Kosten-Komfort-Vergleich).
    # obs_normalized: mit normierten Beobachtungen trainiert (logic/envs.py, normalize=True) —
    # ältere Modelle ohne das Feld bekommen beim Abspielen die Rohwerte, mit denen sie lernten.
    (Path('models') / f'{name}.json').write_text(json.dumps(
        {'algo': algo, 'seed': seed, 'reward': reward, 'timesteps': int(model.num_timesteps),
         'obs_normalized': True}, indent=2))

    if status_path:
        stopped = bool(control_path) and read_control(control_path).get('stop', False)
        write_json(status_path, **status_extra, phase=('gestoppt' if stopped else 'fertig'),
                   timesteps=model.num_timesteps)
    if verbose and not live:
        stats_env = make_env('test', seed=seed, reward_kwargs=reward)
        try:
            action_stats(model, stats_env)
        finally:
            stats_env.close()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--algo', required=True, choices=list(ALGOS))
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--total-timesteps', type=int, default=None, help='überschreibt total_timesteps aus der config')
    ap.add_argument('--csv', default=None, help='Lernkurve live anhängen (für eine Oberfläche)')
    ap.add_argument('--control', default=None, help='Steuerdatei für Pause/Stopp (für eine Oberfläche)')
    ap.add_argument('--status', default=None, help='Statusdatei (für eine Oberfläche)')
    args = ap.parse_args()

    try:
        train(args.algo, args.config, seed=args.seed, total_timesteps_override=args.total_timesteps,
             csv_path=args.csv, control_path=args.control, status_path=args.status)
    except Exception as e:
        if args.status:
            write_json(args.status, phase='fehler', message=str(e))
        raise


if __name__ == '__main__':
    main()
