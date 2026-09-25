"""Qt-Ansicht "Agent": SAC / PPO / TD3 einzeln, zu zweit oder alle drei gleichzeitig
trainieren (Start / Pause / Weiter / Stopp, Netzwerk & Hyperparameter je Algorithmus,
Standard = beste theoretische/Literatur-Empfehlung, live Lernkurve) — und den Agenten auf
der Teststrecke beobachten (Unterreiter, siehe gui/watch_view.py).

Jedes Training läuft als eigener Hintergrundprozess (python -m logic.training), gesteuert
über kleine Dateien in runs/live/ (Details in workbench.md). Fehler in einem Algorithmus
reißen so die anderen nicht mit. Beim Schließen des Fensters wird nachgefragt (stoppen &
sichern / sofort abbrechen), nie still im Hintergrund weitergelaufen; stürzt die Oberfläche
doch einmal ab, findet sie laufende Trainings beim nächsten Start über deren Prozess-ID wieder.
"""
import json
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressDialog, QPushButton,
                               QRadioButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from logic.envs import URL as BOPTEST_URL
from logic.live_control import (ALGOS, MAX_PARALLEL_TRAININGS, RECOMMENDED, anim_path_for, load_yaml_preset,
                                paths_for, read_control, read_status, run_tag, variant_for_w, write_control,
                                write_json)
from gui.compare_view import CompareView
from gui.reward_form import RewardForm
from gui.watch_view import ANIM_PAGE, WatchView, plot_reward_parts

# Zeitfenster (Sekunden), über das die Trainingsgeschwindigkeit für die Restzeit-Schätzung
# gemittelt wird — lang genug gegen Ausreißer, kurz genug, um Tempowechsel mitzubekommen.
RATE_WINDOW_S = 60

PHASE_BADGE = {'wartet': '⏳', 'startet': '🕐', 'läuft': '🟢', 'pausiert': '🟡', 'stoppt': '🟠', 'gestoppt': '🔴',
              'abgebrochen': '⛔', 'fertig': '✅', 'fehler': '⚠️', 'unbekannt': '⚪'}
ACTIVE_PHASES = ('startet', 'läuft', 'pausiert')
# So lange wartet "Stoppen & sichern" beim Schließen des Fensters auf ein sauberes Ende, bevor
# hart beendet wird (normal: wenige Sekunden — aktueller Schritt + Modell speichern).
GRACEFUL_STOP_TIMEOUT_S = 90


class TrainingProcess:
    """Ein Trainingsprozess (python -m logic.training) — frisch gestartet (Popen) oder nach
    einem Neustart der Oberfläche über seine Prozess-ID wiedergefunden. Beenden immer als
    ganzer Prozessbaum: das venv-Python unter Windows ist nur ein Starter, das eigentliche
    Training läuft in einem Kindprozess."""

    def __init__(self, popen=None, pid=None):
        self.popen = popen
        self.pid = popen.pid if popen is not None else pid

    def poll(self):
        """None = läuft noch (wie subprocess.Popen.poll)."""
        if self.popen is not None:
            return self.popen.poll()
        try:
            proc = psutil.Process(self.pid)
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.Error:
            pass
        return 0

    def kill(self):
        try:
            root = psutil.Process(self.pid)
            procs = root.children(recursive=True) + [root]
        except psutil.NoSuchProcess:
            procs = []
        for proc in procs:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(procs, timeout=10)
        if self.popen is not None:
            self.popen.poll()


def release_boptest(testids):
    """BOPTEST-Tests eines hart beendeten Trainings freigeben — der Prozess selbst kann das
    nach einem Kill nicht mehr (siehe logic/training.py::train, finally-Block)."""
    for testid in testids or []:
        try:
            requests.put(f'{BOPTEST_URL}/stop/{testid}', timeout=5)
        except requests.RequestException:
            pass


class HyperparamForm(QGroupBox):
    """Netzwerk & Hyperparameter für einen Algorithmus — vorbelegt mit der besten
    theoretischen Empfehlung (RECOMMENDED), umschaltbar auf die projekt-getunten
    configs/*.yaml-Werte, danach bleibt jedes Feld frei änderbar."""

    def __init__(self, algo: str, root: Path, parent=None):
        super().__init__(f'⚙ {algo} — Netzwerk & Hyperparameter', parent)
        self.algo = algo
        self.root = root
        self._fields = {}

        outer = QVBoxLayout(self)
        self.rb_recommended = QRadioButton('Beste theoretische Empfehlung')
        self.rb_tuned = QRadioButton('Projekt-getuned (configs/*.yaml)')
        self.rb_recommended.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.rb_recommended)
        group.addButton(self.rb_tuned)
        self.rb_recommended.toggled.connect(lambda checked: checked and self._load_preset())
        self.rb_tuned.toggled.connect(lambda checked: checked and self._load_preset())
        outer.addWidget(self.rb_recommended)
        outer.addWidget(self.rb_tuned)

        self.form = QFormLayout()
        outer.addLayout(self.form)
        self._build_fields()
        self._load_preset()

    def _add_line(self, key, label):
        w = QLineEdit()
        self.form.addRow(label, w)
        self._fields[key] = w

    def _add_int(self, key, label, lo, hi, step):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        self.form.addRow(label, w)
        self._fields[key] = w

    def _add_double(self, key, label, lo, hi, step, decimals):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        self.form.addRow(label, w)
        self._fields[key] = w

    def _build_fields(self):
        self._add_line('net_arch', 'Netzwerkgröße (Schichten, kommagetrennt)')
        self._add_int('total_timesteps', 'Zeitschritte gesamt', 1_000, 5_000_000, 1_000)
        self._add_int('eval_freq', 'Eval-Häufigkeit', 100, 100_000, 100)
        self._add_double('learning_rate', 'Lernrate', 0.000001, 1.0, 0.00001, 6)
        self._add_int('batch_size', 'Batch-Größe', 8, 8192, 8)
        self._add_double('gamma', 'Gamma (Diskontierung)', 0.90, 0.999, 0.001, 3)
        if self.algo in ('SAC', 'TD3'):
            self._add_int('buffer_size', 'Replay-Buffer-Größe', 1_000, 2_000_000, 10_000)
            self._add_double('tau', 'Tau (weiches Ziel-Update)', 0.001, 1.0, 0.001, 3)
            self._add_int('train_freq', 'train_freq (Schritte je Update)', 1, 100, 1)
            self._add_int('gradient_steps', 'gradient_steps', 1, 100, 1)
        if self.algo == 'SAC':
            self._add_line('ent_coef', "ent_coef ('auto' oder Zahl)")
        if self.algo == 'TD3':
            self._add_int('policy_delay', 'policy_delay', 1, 20, 1)
            self._add_double('target_policy_noise', 'target_policy_noise', 0.0, 1.0, 0.01, 2)
            self._add_double('target_noise_clip', 'target_noise_clip', 0.0, 1.0, 0.01, 2)
        if self.algo == 'PPO':
            self._add_int('n_steps', 'n_steps (Rollout-Länge)', 32, 100_000, 32)
            self._add_int('n_epochs', 'n_epochs', 1, 100, 1)
            self._add_double('gae_lambda', 'gae_lambda', 0.80, 1.0, 0.01, 2)
            self._add_double('clip_range', 'clip_range', 0.05, 0.5, 0.01, 2)
            self._add_double('ent_coef_ppo', 'ent_coef', 0.0, 1.0, 0.001, 3)

    def _load_preset(self):
        preset = RECOMMENDED[self.algo] if self.rb_recommended.isChecked() else load_yaml_preset(self.algo, self.root)
        p = preset['params']
        self._fields['net_arch'].setText(','.join(str(n) for n in preset['net_arch']))
        self._fields['total_timesteps'].setValue(int(preset['total_timesteps']))
        self._fields['eval_freq'].setValue(int(preset['eval_freq']))
        self._fields['learning_rate'].setValue(float(p['learning_rate']))
        self._fields['batch_size'].setValue(int(p['batch_size']))
        self._fields['gamma'].setValue(float(p['gamma']))
        if self.algo in ('SAC', 'TD3'):
            self._fields['buffer_size'].setValue(int(p['buffer_size']))
            self._fields['tau'].setValue(float(p['tau']))
            self._fields['train_freq'].setValue(int(p['train_freq']))
            self._fields['gradient_steps'].setValue(int(p['gradient_steps']))
        if self.algo == 'SAC':
            self._fields['ent_coef'].setText(str(p['ent_coef']))
        if self.algo == 'TD3':
            self._fields['policy_delay'].setValue(int(p['policy_delay']))
            self._fields['target_policy_noise'].setValue(float(p['target_policy_noise']))
            self._fields['target_noise_clip'].setValue(float(p['target_noise_clip']))
        if self.algo == 'PPO':
            self._fields['n_steps'].setValue(int(p['n_steps']))
            self._fields['n_epochs'].setValue(int(p['n_epochs']))
            self._fields['gae_lambda'].setValue(float(p['gae_lambda']))
            self._fields['clip_range'].setValue(float(p['clip_range']))
            self._fields['ent_coef_ppo'].setValue(float(p['ent_coef']))

    def values(self) -> dict:
        try:
            net_arch = [int(x) for x in self._fields['net_arch'].text().split(',') if x.strip()]
        except ValueError:
            net_arch = RECOMMENDED[self.algo]['net_arch']
        params = {'learning_rate': self._fields['learning_rate'].value(),
                  'batch_size': self._fields['batch_size'].value(), 'gamma': self._fields['gamma'].value()}
        if self.algo in ('SAC', 'TD3'):
            params['buffer_size'] = self._fields['buffer_size'].value()
            params['tau'] = self._fields['tau'].value()
            params['train_freq'] = self._fields['train_freq'].value()
            params['gradient_steps'] = self._fields['gradient_steps'].value()
        if self.algo == 'SAC':
            ent = self._fields['ent_coef'].text()
            try:
                params['ent_coef'] = float(ent)
            except ValueError:
                params['ent_coef'] = ent
        if self.algo == 'TD3':
            params['policy_delay'] = self._fields['policy_delay'].value()
            params['target_policy_noise'] = self._fields['target_policy_noise'].value()
            params['target_noise_clip'] = self._fields['target_noise_clip'].value()
        if self.algo == 'PPO':
            params['n_steps'] = self._fields['n_steps'].value()
            params['n_epochs'] = self._fields['n_epochs'].value()
            params['gae_lambda'] = self._fields['gae_lambda'].value()
            params['clip_range'] = self._fields['clip_range'].value()
            params['ent_coef'] = self._fields['ent_coef_ppo'].value()
        return {'algo': self.algo, 'total_timesteps': self._fields['total_timesteps'].value(),
               'eval_freq': self._fields['eval_freq'].value(), 'net_arch': net_arch, 'params': params}


class AgentView(QWidget):
    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = root
        self.models_dir = root / 'models'
        # Lauf-Name (z. B. 'sac_seed0' oder 'sac_seed0_w0_3') -> {'proc' (None = wartet noch),
        # 'queued', 'cmd', 'algo', 'label', 'seed', 'paths', 'logfile', 'eval_freq', 'samples'}
        self.jobs: dict[str, dict] = {}
        self._study = None   # laufende Komfortgewicht-Studie: {'runs': [...], 'models': [...]}
        self._known_models: set[str] = set()
        self._build_ui()
        self._reattach_running()
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self._refresh_panel)
        self.timer.start()

    def _build_ui(self):
        outer = QHBoxLayout(self)

        left = QWidget()
        left.setFixedWidth(400)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel('<b>Algorithmen</b> — einzeln, zu zweit oder alle drei'))
        self.algo_checks = {a: QCheckBox(a) for a in ALGOS}
        self.algo_checks['SAC'].setChecked(True)
        for cb in self.algo_checks.values():
            left_layout.addWidget(cb)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel('Seed:'))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 100_000)
        seed_row.addWidget(self.seed_spin)
        left_layout.addLayout(seed_row)

        btn_row1 = QHBoxLayout()
        self.start_btn = QPushButton('▶ Start')
        self.stop_btn = QPushButton('⏹ Stopp')
        btn_row1.addWidget(self.start_btn); btn_row1.addWidget(self.stop_btn)
        btn_row2 = QHBoxLayout()
        self.pause_btn = QPushButton('⏸ Pause')
        self.resume_btn = QPushButton('⏵ Weiter')
        btn_row2.addWidget(self.pause_btn); btn_row2.addWidget(self.resume_btn)
        self.force_btn = QPushButton('⛔ Sofort erzwingen (ohne Sicherung)')
        self.start_btn.setProperty('accent', True)
        self.stop_btn.setProperty('danger', True)
        self.force_btn.setProperty('danger', True)
        left_layout.addLayout(btn_row1)
        left_layout.addLayout(btn_row2)
        left_layout.addWidget(self.force_btn)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.pause_btn.clicked.connect(lambda: self._set_pause(True))
        self.resume_btn.clicked.connect(lambda: self._set_pause(False))
        self.force_btn.clicked.connect(self._on_force)

        left_layout.addWidget(QLabel('<b>Belohnung & Hyperparameter</b>'))
        self.reward_form = RewardForm()
        self.forms = {a: HyperparamForm(a, self.root) for a in ALGOS}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        forms_widget = QWidget()
        forms_layout = QVBoxLayout(forms_widget)
        forms_layout.addWidget(self.reward_form)
        for a in ALGOS:
            forms_layout.addWidget(self.forms[a])
        forms_layout.addStretch()
        scroll.setWidget(forms_widget)
        left_layout.addWidget(scroll, 1)

        sub_tabs = QTabWidget()
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        self.status_label = QLabel('Noch kein Training gestartet. Links Algorithmus/-en wählen und „▶ Start" drücken.')
        self.status_label.setWordWrap(True)
        train_layout.addWidget(self.status_label)
        self.figure = Figure(figsize=(7, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        # Oben die Animation des laufenden Trainings, unten die Lernkurve — beide gleichzeitig,
        # Trennlinie verschiebbar.
        self.train_anim = QWebEngineView()
        anim_url = QUrl.fromLocalFile(str(ANIM_PAGE))
        anim_url.setQuery('embedded=1')
        self.train_anim.setUrl(anim_url)
        self.train_anim.loadFinished.connect(lambda _ok: self._push_train_anim(force=True))
        self._anim_sent = None
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.train_anim)
        splitter.addWidget(self.canvas)
        splitter.setSizes([420, 480])
        train_layout.addWidget(splitter, 1)
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._push_train_anim)
        self.anim_timer.start(600)
        sub_tabs.addTab(train_tab, 'Training')

        self.watch_view = WatchView(self.models_dir)
        sub_tabs.addTab(self.watch_view, 'Beobachten')
        self.compare_view = CompareView(self.models_dir, self.root / 'results' / 'compare.csv',
                                        can_run=self._compare_allowed, start_study=self.start_study)
        sub_tabs.addTab(self.compare_view, 'Vergleich')
        sub_tabs.currentChanged.connect(lambda i: i == 1 and self.watch_view.reload_models())

        outer.addWidget(left)
        outer.addWidget(sub_tabs, 1)

    # ---------- Trainingssteuerung ----------

    def _running_jobs(self):
        return {t: j for t, j in self.jobs.items() if j['proc'] is not None and j['proc'].poll() is None}

    def _queued_jobs(self):
        return {t: j for t, j in self.jobs.items() if j.get('queued')}

    @staticmethod
    def _label(algo: str, variant: str | None) -> str:
        if not variant:
            return algo
        return f'{algo} · ' + (f'w={variant[1:].replace("_", ".")}' if variant.startswith('w') else variant)

    def _compare_allowed(self) -> tuple[bool, str]:
        """Der Vergleich braucht selbst einen BOPTEST-Worker — nicht starten, wenn alle belegt sind."""
        if len(self._running_jobs()) >= MAX_PARALLEL_TRAININGS:
            return False, (f'Alle BOPTEST-Worker sind gerade durch {MAX_PARALLEL_TRAININGS} Trainings belegt. '
                           'Bitte warten, bis eines fertig ist.')
        return True, ''

    def _on_start(self):
        selected = [a for a, cb in self.algo_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, 'Kein Algorithmus gewählt', 'Bitte mindestens einen Algorithmus auswählen.')
            return
        self._queue_runs(selected, [(None, self.reward_form.values())])
        self._start_queued()
        self._refresh_panel()

    def _queue_runs(self, algos, variants) -> list[str]:
        """Je Algorithmus und Variante (Name oder None, Belohnungsparameter) einen Lauf in die
        Warteschlange stellen. Hyperparameter aus den Formularen links. Gibt die Lauf-Namen zurück."""
        seed = self.seed_spin.value()
        tags = []
        for algo in algos:
            hp = self.forms[algo].values()
            if algo == 'PPO' and hp['params']['n_steps'] % hp['params']['batch_size'] != 0:
                QMessageBox.warning(self, 'PPO nicht gestartet', 'n_steps muss durch batch_size teilbar sein.')
                continue
            for variant, rkw in variants:
                tag = run_tag(algo, seed, variant)
                existing = self.jobs.get(tag)
                if existing and (existing.get('queued') or tag in self._running_jobs()):
                    tags.append(tag)
                    continue  # läuft schon / wartet schon
                paths = paths_for(algo, seed, self.root, variant)
                write_control(paths['control'], pause=False, stop=False)
                run_cfg = {'algo': algo, 'name': tag, 'total_timesteps': hp['total_timesteps'],
                          'eval_freq': hp['eval_freq'], 'reward': rkw, 'params': hp['params'],
                          'net_arch': hp['net_arch']}
                paths['run_config'].write_text(json.dumps(run_cfg, indent=2))
                if paths['csv'].exists():
                    paths['csv'].unlink()   # alte Lernkurve eines früheren Laufs gleichen Namens
                write_json(paths['status'], phase='wartet', timesteps=0)
                cmd = [sys.executable, '-m', 'logic.training',
                      '--algo', algo, '--config', str(paths['run_config']), '--seed', str(seed),
                      '--csv', str(paths['csv']), '--control', str(paths['control']),
                      '--status', str(paths['status'])]
                self.jobs[tag] = {'proc': None, 'queued': True, 'cmd': cmd, 'algo': algo,
                                  'label': self._label(algo, variant), 'seed': seed, 'paths': paths,
                                  'logfile': None, 'eval_freq': hp['eval_freq'], 'samples': deque()}
                tags.append(tag)
        return tags

    def start_study(self, weights: list[float], algos: list[str], skip_existing: bool) -> dict:
        """Komfortgewicht-Studie (aus dem Reiter "Vergleich"): je Algorithmus und w ein Lauf mit
        sonst gleicher Belohnung (Formular links). skip_existing: w-Werte, für die schon ein
        Modell existiert, nicht neu trainieren. Nach dem letzten Lauf vergleicht der Reiter
        "Vergleich" automatisch (siehe _refresh_panel)."""
        seed = self.seed_spin.value()
        reward_kwargs = self.reward_form.values()
        to_train, reused = [], []
        for algo in algos:
            for w in weights:
                variant = variant_for_w(w)
                tag = run_tag(algo, seed, variant)
                if skip_existing and (self.models_dir / f'{tag}.zip').exists():
                    reused.append(tag)
                else:
                    to_train.append((algo, variant, {**reward_kwargs, 'w_comfort': w}))
        started = []
        for algo, variant, rkw in to_train:
            started += self._queue_runs([algo], [(variant, rkw)])
        self._study = {'runs': started, 'models': started + reused}
        self._start_queued()
        self._refresh_panel()
        return {'started': started, 'reused': reused}

    def _study_status(self):
        """Fortschritt der laufenden Studie an den Reiter "Vergleich" melden; ist alles fertig,
        dort automatisch vergleichen."""
        study = getattr(self, '_study', None)
        if not study:
            return
        active = self._running_jobs().keys() | self._queued_jobs().keys()
        open_runs = [t for t in study['runs'] if t in active]
        done = len(study['runs']) - len(open_runs)
        if open_runs:
            self.compare_view.set_study_status(
                f'Studie: {done} von {len(study["runs"])} Trainings fertig — Fortschritt im Reiter '
                '„Training“. Danach wird automatisch verglichen.')
            return
        self._study = None
        self.compare_view.study_finished(study['models'])

    def _start_queued(self):
        """Wartende Läufe starten, solange BOPTEST-Worker frei sind (je Training zwei: Training +
        Auswertung). Weitere warten und starten automatisch, sobald ein Lauf endet."""
        for job in self._queued_jobs().values():
            if len(self._running_jobs()) >= MAX_PARALLEL_TRAININGS:
                break
            job['logfile'] = open(job['paths']['log'], 'w')
            popen = subprocess.Popen(job['cmd'], cwd=str(self.root), stdout=job['logfile'],
                                     stderr=subprocess.STDOUT)
            job['proc'] = TrainingProcess(popen=popen)
            job['queued'] = False

    def _cancel_queued(self):
        for job in self._queued_jobs().values():
            job['queued'] = False
            write_json(job['paths']['status'], phase='gestoppt', timesteps=0)

    def _set_pause(self, flag: bool):
        for job in self._running_jobs().values():
            ctrl = read_control(job['paths']['control'])
            write_control(job['paths']['control'], pause=flag, stop=ctrl.get('stop', False))

    def _on_stop(self):
        """Sauber stoppen: das Training beendet den aktuellen Schritt (bzw. bricht eine
        laufende Auswertung ab), sichert das Modell, gibt BOPTEST frei und beendet sich.
        Noch wartende Läufe werden gar nicht erst gestartet."""
        self._cancel_queued()
        for job in self._running_jobs().values():
            write_control(job['paths']['control'], pause=False, stop=True)
            job['stopping'] = True
        self._refresh_panel()

    def _on_force(self):
        running = self._running_jobs()
        if not running and not self._queued_jobs():
            return
        answer = QMessageBox.question(
            self, 'Sofort abbrechen?',
            'Das Training wird sofort beendet, ohne den aktuellen Stand zu sichern. Bereits '
            'gesicherte Zwischenmodelle (beste Auswertung) bleiben erhalten.\n\nFortfahren?')
        if answer == QMessageBox.StandardButton.Yes:
            self._cancel_queued()
            self._kill_jobs(running)

    def _kill_jobs(self, jobs: dict):
        for job in jobs.values():
            job['proc'].kill()
            status = read_status(job['paths']['status'])
            release_boptest(status.get('testids'))
            write_json(job['paths']['status'], phase='abgebrochen', timesteps=status.get('timesteps', 0))
            job['stopping'] = False
        self._refresh_panel()

    def confirm_close(self) -> bool:
        """Vom Hauptfenster beim Schließen aufgerufen. False = Fenster offen lassen. Laufende
        Trainings werden nie stillschweigend im Hintergrund zurückgelassen."""
        running = self._running_jobs()
        if not running:
            self._cancel_queued()
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle('Training läuft noch')
        names = ', '.join(j['label'] for j in running.values())
        queued = len(self._queued_jobs())
        box.setText(f'Es läuft noch ein Training ({names}). Was soll damit passieren?')
        box.setInformativeText('„Stoppen & sichern“ beendet den aktuellen Schritt, speichert das '
                               'Modell und gibt BOPTEST frei (dauert meist nur Sekunden).'
                               + (f' {queued} wartende Läufe werden nicht mehr gestartet.' if queued else ''))
        graceful = box.addButton('Stoppen && sichern', QMessageBox.ButtonRole.AcceptRole)
        force = box.addButton('Sofort abbrechen', QMessageBox.ButtonRole.DestructiveRole)
        box.addButton('Fenster offen lassen', QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(graceful)
        box.exec()
        if box.clickedButton() is force:
            self._cancel_queued()
            self._kill_jobs(running)
            return True
        if box.clickedButton() is not graceful:
            return False

        self._on_stop()
        dialog = QProgressDialog('Training wird gestoppt, Modell wird gesichert …', None, 0, 0, self)
        dialog.setWindowTitle('Bitte warten')
        dialog.setMinimumDuration(0)
        dialog.show()
        deadline = time.time() + GRACEFUL_STOP_TIMEOUT_S
        while self._running_jobs() and time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.1)
        dialog.close()
        leftover = self._running_jobs()
        if leftover:   # hängt (z. B. BOPTEST antwortet nicht mehr) -> hart beenden
            self._kill_jobs(leftover)
        return True

    def _reattach_running(self):
        """Trainings wiederfinden, die noch aus einer früheren Sitzung laufen (z. B. nach einem
        Absturz der Oberfläche) — damit Status, Lernkurve, Pause und Stopp wieder funktionieren."""
        live_dir = self.root / 'runs' / 'live'
        for status_path in sorted(live_dir.glob('*.status.json')) if live_dir.exists() else []:
            status = read_status(status_path)
            tag = status_path.name[:-len('.status.json')]
            m = re.fullmatch(r'(sac|ppo|td3)_seed(\d+)(?:_(.+))?', tag)
            pid = status.get('pid')
            if not m or not pid or status.get('phase') not in ACTIVE_PHASES:
                continue
            try:
                cmdline = ' '.join(psutil.Process(pid).cmdline())
            except psutil.Error:
                continue
            if 'logic.training' not in cmdline or status_path.name not in cmdline:
                continue   # Prozess-ID inzwischen von einem anderen Programm belegt
            algo, seed, variant = m.group(1).upper(), int(m.group(2)), m.group(3)
            paths = paths_for(algo, seed, self.root, variant)
            try:
                eval_freq = json.loads(paths['run_config'].read_text()).get('eval_freq')
            except (OSError, ValueError):
                eval_freq = None
            self.jobs[tag] = {'proc': TrainingProcess(pid=pid), 'queued': False, 'algo': algo,
                              'label': self._label(algo, variant), 'seed': seed, 'paths': paths,
                              'logfile': None, 'eval_freq': eval_freq, 'samples': deque()}

    @staticmethod
    def _next_eval_text(job: dict, phase: str, steps: int) -> str:
        """'nächste Auswertung bei 2.000 Schritten — noch ca. 12 min' aus der gemessenen
        Geschwindigkeit (Schritte je Sekunde über die letzten RATE_WINDOW_S Sekunden)."""
        eval_freq = job.get('eval_freq')
        if not eval_freq or phase not in ('startet', 'läuft', 'pausiert'):
            return ''
        next_eval = (steps // eval_freq + 1) * eval_freq
        target = f'nächste Auswertung bei {next_eval:,} Schritten'.replace(',', '.')
        if phase == 'pausiert':
            return f'{target} (pausiert)'

        now = time.time()
        samples = job['samples']
        if not samples or samples[-1][1] != steps:
            samples.append((now, steps))
        while len(samples) > 2 and now - samples[0][0] > RATE_WINDOW_S:
            samples.popleft()
        (t0, s0), (t1, s1) = samples[0], samples[-1]
        if s1 <= s0 or t1 - t0 < 5:
            return f'{target} — Geschwindigkeit wird gemessen …'
        rate = (s1 - s0) / (t1 - t0)
        minutes = (next_eval - steps) / rate / 60
        eta = 'unter 1 min' if minutes < 1 else f'ca. {minutes:.0f} min'
        return f'{target} — noch {eta} ({rate:.1f} Schritte/s)'

    def _push_train_anim(self, force: bool = False):
        """Neuesten Animationszustand des laufenden (sonst zuletzt gestarteten) Laufs an die
        Animation oben im Training-Reiter geben — geschrieben von
        logic/training.py::LiveCallback, beim Training wie bei den Auswertungen."""
        jobs = getattr(self, 'jobs', {})
        candidates = list(self._running_jobs().values()) or [j for j in jobs.values() if not j.get('queued')]
        if not candidates:
            return
        job = candidates[-1]
        try:
            text = anim_path_for(job['paths']['status']).read_text(encoding='utf-8')
            state = json.loads(text)
        except (OSError, ValueError):   # noch nichts geschrieben oder gerade ersetzt
            return
        if not force and text == self._anim_sent:
            return
        self._anim_sent = text
        state['strategy'] = f"{job['label']} · {state.get('strategy', '')}"
        self.train_anim.page().runJavaScript(f'window.applyState && window.applyState({json.dumps(state)})')

    def _refresh_panel(self):
        self._start_queued()
        self._study_status()
        running = bool(self._running_jobs())
        active = running or bool(self._queued_jobs())
        for btn in (self.pause_btn, self.resume_btn):
            btn.setEnabled(running)
        for btn in (self.stop_btn, self.force_btn):
            btn.setEnabled(active)
        if not self.jobs:
            return
        lines = []
        pending = []   # Restzeit-Hinweise für den Platzhalter, solange noch keine Kurve da ist
        curves = []    # (Beschriftung, Lernkurven-DataFrame) je Lauf mit mindestens einer Auswertung
        for job in self.jobs.values():
            label = job['label']
            status = read_status(job['paths']['status'])
            phase = status.get('phase', 'unbekannt')
            alive = job['proc'] is not None and job['proc'].poll() is None
            if job.get('queued'):
                phase = 'wartet'
            elif not alive and phase in ACTIVE_PHASES:
                phase = 'abgebrochen'   # Prozess weg, ohne Endstatus zu schreiben
            elif alive and job.get('stopping'):
                phase = 'stoppt'
            if not alive and job.get('logfile'):
                job['logfile'].close()
                job['logfile'] = None
            badge = PHASE_BADGE.get(phase, '⚪')
            steps = status.get('timesteps', 0)
            reward_txt = f", Ø Belohnung {status['reward']:.2f}" if 'reward' in status else ''
            phase_txt = {'wartet': ' — wartet auf freie BOPTEST-Worker',
                         'stoppt': ' — wird gestoppt, Modell wird gesichert …',
                         'abgebrochen': ' — abgebrochen', 'gestoppt': ' — gestoppt',
                         'pausiert': ' — pausiert'}.get(phase, '')
            if phase == 'läuft' and status.get('evaluating'):
                phase_txt = ' — Auswertung läuft …'
            lines.append(f"{badge} {label} (seed {job['seed']}): {steps:,} Schritte{reward_txt}".replace(',', '.')
                         + phase_txt)
            if phase == 'fehler' and 'message' in status:
                lines.append(f"    ⚠️ {status['message']}")
            eta = self._next_eval_text(job, phase, steps)
            if eta:
                lines.append(f"    ⏱ {eta}")
                pending.append(f"{label}: {eta}")

            csv_path = job['paths']['csv']
            if csv_path.exists():
                try:
                    d = pd.read_csv(csv_path)
                except Exception:
                    d = pd.DataFrame()
                if not d.empty:
                    curves.append((label, d))

        self.status_label.setText('\n'.join(lines))
        self.figure.clear()
        # Unter der Lernkurve je Lauf eine Reihe mit den Belohnungsanteilen je Auswertung
        # (Spalten reward_cost … aus logic/training.py::LiveCallback; ältere Läufe haben sie nicht).
        with_parts = [(label, d) for label, d in curves if 'reward_cost' in d]
        if with_parts:
            axes = self.figure.subplots(1 + len(with_parts), 1, sharex=True,
                                        height_ratios=[2] + [1] * len(with_parts))
            ax = axes[0]
        else:
            ax = self.figure.add_subplot(111)
        if curves:
            for label, d in curves:
                ax.plot(d.timesteps, d.reward, marker='o', label=label)
            ax.set_ylabel('Ø Belohnung (Testzeitraum)')
            ax.legend()
            for part_ax, (label, d) in zip(axes[1:] if with_parts else [], with_parts):
                steps = d.timesteps.to_numpy(dtype=float)
                gap = np.diff(steps).min() if len(steps) > 1 else steps[0]
                plot_reward_parts(part_ax, steps, d, width=0.7 * gap, title=label, clip_terminal=False)
                part_ax.set_ylabel('Anteile', fontsize=8)
            (axes[-1] if with_parts else ax).set_xlabel('Zeitschritte')
        else:
            text = 'Lernkurve erscheint hier, sobald die erste Auswertung durchgelaufen ist.'
            if pending:
                text += '\n\n' + '\n'.join(pending)
            ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, fontsize=9,
                   color='#8A98A3', linespacing=1.6)
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

        # Neue/bessere Modelle könnten gerade erst gesichert worden sein — die "Beobachten"-
        # Ansicht nachziehen, aber nur bei tatsächlicher Änderung (sonst würde alle 2s die
        # laufende Wiedergabe/Auswahl dort zurückgesetzt).
        current_models = {p.stem for p in self.models_dir.glob('*.zip')} if self.models_dir.exists() else set()
        if current_models != self._known_models:
            self._known_models = current_models
            self.watch_view.reload_models()
            self.compare_view.mark_outdated()
