"""Qt-Ansicht "Agent": SAC / PPO / TD3 einzeln, zu zweit oder alle drei gleichzeitig
trainieren (Start / Pause / Weiter / Stopp, Netzwerk & Hyperparameter je Algorithmus,
Standard = beste theoretische/Literatur-Empfehlung, live Lernkurve) — und den Agenten auf
der Teststrecke beobachten (Unterreiter, siehe gui/watch_view.py).

Jedes Training läuft als eigener Hintergrundprozess (python -m logic.training), gesteuert
über kleine Dateien in runs/live/ (Details in workbench.md). Das hält die Sache robust: das
Training läuft weiter, auch wenn das Fenster neu geladen wird, und Fehler in einem
Algorithmus reißen die anderen nicht mit.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton,
                               QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from logic.live_control import (ALGOS, RECOMMENDED, load_yaml_preset, paths_for, read_control,
                                read_status, write_control)
from gui.reward_form import RewardForm
from gui.watch_view import WatchView

PHASE_BADGE = {'startet': '🕐', 'läuft': '🟢', 'pausiert': '🟡', 'gestoppt': '🔴',
              'fertig': '✅', 'fehler': '⚠️', 'unbekannt': '⚪'}


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
        self.jobs: dict[str, dict] = {}   # algo -> {'proc', 'seed', 'paths', 'logfile'}
        self._known_models: set[str] = set()
        self._build_ui()
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
        train_layout.addWidget(self.canvas, 1)
        sub_tabs.addTab(train_tab, 'Training')

        self.watch_view = WatchView(self.models_dir)
        sub_tabs.addTab(self.watch_view, 'Beobachten')
        sub_tabs.currentChanged.connect(lambda i: i == 1 and self.watch_view.reload_models())

        outer.addWidget(left)
        outer.addWidget(sub_tabs, 1)

    # ---------- Trainingssteuerung ----------

    def _running_jobs(self):
        return {a: j for a, j in self.jobs.items() if j['proc'].poll() is None}

    def _on_start(self):
        selected = [a for a, cb in self.algo_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, 'Kein Algorithmus gewählt', 'Bitte mindestens einen Algorithmus auswählen.')
            return
        seed = self.seed_spin.value()
        reward_kwargs = self.reward_form.values()
        for algo in selected:
            running = self.jobs.get(algo)
            if running and running['proc'].poll() is None:
                continue  # läuft schon
            hp = self.forms[algo].values()
            if algo == 'PPO' and hp['params']['n_steps'] % hp['params']['batch_size'] != 0:
                QMessageBox.warning(self, 'PPO nicht gestartet', 'n_steps muss durch batch_size teilbar sein.')
                continue
            paths = paths_for(algo, seed, self.root)
            write_control(paths['control'], pause=False, stop=False)
            run_cfg = {'algo': algo, 'total_timesteps': hp['total_timesteps'], 'eval_freq': hp['eval_freq'],
                      'reward': reward_kwargs, 'params': hp['params'], 'net_arch': hp['net_arch']}
            paths['run_config'].write_text(json.dumps(run_cfg, indent=2))
            cmd = [sys.executable, '-m', 'logic.training',
                  '--algo', algo, '--config', str(paths['run_config']), '--seed', str(seed),
                  '--csv', str(paths['csv']), '--control', str(paths['control']), '--status', str(paths['status'])]
            logfile = open(paths['log'], 'w')
            proc = subprocess.Popen(cmd, cwd=str(self.root), stdout=logfile, stderr=subprocess.STDOUT)
            self.jobs[algo] = {'proc': proc, 'seed': seed, 'paths': paths, 'logfile': logfile}
        self._refresh_panel()

    def _set_pause(self, flag: bool):
        for job in self._running_jobs().values():
            ctrl = read_control(job['paths']['control'])
            write_control(job['paths']['control'], pause=flag, stop=ctrl.get('stop', False))

    def _on_stop(self):
        for job in self._running_jobs().values():
            write_control(job['paths']['control'], pause=False, stop=True)

    def _on_force(self):
        for job in self._running_jobs().values():
            job['proc'].terminate()

    def _refresh_panel(self):
        if not self.jobs:
            return
        lines = []
        any_data = False
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for algo, job in self.jobs.items():
            status = read_status(job['paths']['status'])
            phase = status.get('phase', 'unbekannt')
            badge = PHASE_BADGE.get(phase, '⚪')
            steps = status.get('timesteps', 0)
            reward_txt = f", Ø Belohnung {status['reward']:.2f}" if 'reward' in status else ''
            lines.append(f"{badge} {algo} (seed {job['seed']}): {steps:,} Schritte{reward_txt}".replace(',', '.'))
            if phase == 'fehler' and 'message' in status:
                lines.append(f"    ⚠️ {status['message']}")

            csv_path = job['paths']['csv']
            if csv_path.exists():
                try:
                    d = pd.read_csv(csv_path)
                except Exception:
                    d = pd.DataFrame()
                if not d.empty:
                    any_data = True
                    ax.plot(d.timesteps, d.reward, marker='o', label=algo)

        self.status_label.setText('\n'.join(lines))
        if any_data:
            ax.set_xlabel('Zeitschritte')
            ax.set_ylabel('Ø Belohnung (Testzeitraum)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Lernkurve erscheint hier, sobald die erste Auswertung durchgelaufen ist.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=9, color='#8A98A3')
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

        # Neue/bessere Modelle könnten gerade erst gesichert worden sein — die "Beobachten"-
        # Ansicht nachziehen, aber nur bei tatsächlicher Änderung (sonst würde alle 2s die
        # laufende Wiedergabe/Auswahl dort zurückgesetzt).
        current_models = {p.stem for p in self.models_dir.glob('*.zip')} if self.models_dir.exists() else set()
        if current_models != self._known_models:
            self._known_models = current_models
            self.watch_view.reload_models()
