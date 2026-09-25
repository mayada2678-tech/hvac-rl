"""Qt-Ansicht "Vergleich": trainierte Modelle und RBC auf derselben Testperiode nebeneinander —
mit frei wählbaren Kennzahlen auf beiden Achsen, frei wählbarer Referenz (RBC oder ein
beliebiges Modell) und an-/abwählbaren Modellen.

Typischer Einsatz: Komfortgewicht-Studie (Reiter "Training", Belohnungsformular), dann hier
Stromkosten über Komfortverletzung legen und die Kosten-Komfort-Kurve ablesen:
„Bei w=1 kaum Ersparnis, bei w=0.1 sparen wir X %, aber Y Kelvin-Stunden Komfortverletzung“.
Genauso lässt sich aber z. B. CO₂ über Netzbezug oder Ersparnis über w auftragen.

Die Simulation läuft in einem Hintergrund-Thread (je Modell eine Testepisode, ~0.5-1 min),
Ergebnisse werden in results/compare.csv zwischengespeichert (logic/evaluation.py::compare).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from logic.evaluation import METRICS, RBC, compare, display_name, savings_vs, summary_sentences
from logic.live_control import ALGOS, DEFAULT_STUDY_WEIGHTS, parse_weights, run_tag, variant_for_w

# Farbe UND Markerform je Algorithmus (Identität nie nur über Farbe), fest je Algorithmus —
# nicht nach Reihenfolge vergeben. Gleiche Farben wie im Reiter "Beobachten".
ALGO_STYLE = {'SAC': ('#4FC1FF', 'o'), 'PPO': ('#4EC9B0', 's'), 'TD3': ('#F48771', '^')}
RBC_STYLE = ('#DCDCAA', '*')
# Beschriftungs-Versatz je Algorithmus (Punkte), damit sich die Labels verschiedener
# Algorithmen am selben Ort (typisch: alle w=1 nahe beim RBC) nicht überdecken.
LABEL_OFFSET = {'SAC': (8, -13), 'TD3': (8, 7), 'PPO': (-8, 7)}
TEXT = '#cccccc'
MUTED = '#8A98A3'

DEFAULT_X, DEFAULT_Y = 'tdis_kh', 'cost_eur'   # die klassische Kosten-Komfort-Kurve


def fmt(key: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    if key == 'savings_pct':
        return f'{v:+.1f} %'
    if key == 'w':
        return f'{v:g}'
    return f'{v:.3g}' if abs(v) < 1 else f'{v:,.2f}'.replace(',', ' ')


class CompareWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, models_dir: Path, cache_path: Path, force: bool):
        super().__init__()
        self.models_dir, self.cache_path, self.force = models_dir, cache_path, force

    def run(self):
        try:
            df = compare(self.models_dir, self.cache_path, force=self.force,
                         progress=lambda i, n, name: self.progress.emit(i, n, name))
            self.finished_ok.emit(df)
        except Exception as e:   # z. B. BOPTEST nicht erreichbar
            self.failed.emit(str(e))


class CompareView(QWidget):
    def __init__(self, models_dir: Path, cache_path: Path, can_run=None, start_study=None, parent=None):
        """start_study(weights, algos, skip_existing) -> {'started': [...], 'reused': [...]}:
        vom Reiter "Agent" bereitgestellt (dort liegen Trainingsprozesse und Hyperparameter)."""
        super().__init__(parent)
        self.start_study = start_study
        self._pending_select = None   # nach dem nächsten Vergleich nur diese Modelle anhaken
        self.models_dir = Path(models_dir)
        self.cache_path = Path(cache_path)
        self.can_run = can_run or (lambda: (True, ''))
        self.worker = None
        self.df = pd.DataFrame()
        self._points = []   # (x, y, Tooltip-Text) für den Hover
        self._ax = self._hover = None
        # Zuletzt bewusst gewählte Achsen/Referenz — nicht die Notlösung, die bei leeren Daten
        # (nur "Ersparnis" verfügbar) in den Auswahlfeldern steht.
        self._choice = {'x': DEFAULT_X, 'y': DEFAULT_Y, 'ref': RBC}
        self._build_ui()
        self._load_cached()

    # ---------- Aufbau ----------

    def _build_ui(self):
        outer = QVBoxLayout(self)

        intro = QLabel('Alle trainierten Modelle und der RBC auf derselben Testperiode. Achsen, Referenz '
                       'und Modelle sind frei wählbar.')
        intro.setWordWrap(True)
        outer.addWidget(intro)
        outer.addWidget(self._build_study_box())

        row = QHBoxLayout()
        self.run_btn = QPushButton('▶ Vergleich berechnen')
        self.run_btn.setProperty('accent', True)
        self.run_btn.setToolTip('Simuliert nur neue oder neu trainierte Modelle; der Rest kommt aus '
                                'results/compare.csv.')
        self.force_btn = QPushButton('↻ Alles neu berechnen')
        self.run_btn.clicked.connect(lambda: self._run(force=False))
        self.force_btn.clicked.connect(lambda: self._run(force=True))
        self.state_label = QLabel()
        self.state_label.setStyleSheet(f'color: {MUTED};')
        self.state_label.setWordWrap(True)
        row.addWidget(self.run_btn)
        row.addWidget(self.force_btn)
        row.addWidget(self.state_label, 1)
        outer.addLayout(row)

        # Was wird verglichen: beide Achsen und die Referenz frei wählbar
        pick = QHBoxLayout()
        self.x_combo, self.y_combo, self.ref_combo = QComboBox(), QComboBox(), QComboBox()
        for label, combo, tip in (
                ('x-Achse:', self.x_combo, 'Kennzahl auf der waagerechten Achse.'),
                ('y-Achse:', self.y_combo, 'Kennzahl auf der senkrechten Achse.'),
                ('Referenz:', self.ref_combo, 'Bezugspunkt für „Ersparnis ggü. Referenz“, die '
                                              'gestrichelten Linien und die Sätze oben.')):
            combo.setToolTip(tip)
            combo.setMinimumContentsLength(14)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.currentIndexChanged.connect(self._on_choice_changed)
            pick.addWidget(QLabel(label))
            pick.addWidget(combo, 1)
        self.swap_btn = QPushButton('⇄')
        self.swap_btn.setToolTip('Achsen tauschen')
        self.swap_btn.setFixedWidth(36)
        self.swap_btn.clicked.connect(self._swap_axes)
        pick.insertWidget(2, self.swap_btn)   # zwischen x- und y-Auswahl
        outer.addLayout(pick)

        summary_row = QHBoxLayout()
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.copy_btn = QPushButton('📋 Kopieren')
        self.copy_btn.setToolTip('Ergebnis in Worten in die Zwischenablage kopieren.')
        self.copy_btn.clicked.connect(self._copy_summary)
        summary_row.addWidget(self.summary, 1)
        summary_row.addWidget(self.copy_btn, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(summary_row)

        # Diagramm | Modellauswahl, darunter die Tabelle
        top = QSplitter(Qt.Orientation.Horizontal)
        self.figure = Figure(figsize=(7, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        top.addWidget(self.canvas)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(QLabel('<b>Modelle</b>'))
        self.model_list = QListWidget()
        self.model_list.setToolTip('Haken entfernen = Modell aus Diagramm, Tabelle und Sätzen ausblenden.')
        self.model_list.itemChanged.connect(lambda _item: self._render())
        side_layout.addWidget(self.model_list, 1)
        sel_row = QHBoxLayout()
        all_btn, none_btn = QPushButton('Alle'), QPushButton('Keine')
        all_btn.clicked.connect(lambda: self._check_all(True))
        none_btn.clicked.connect(lambda: self._check_all(False))
        sel_row.addWidget(all_btn)
        sel_row.addWidget(none_btn)
        side_layout.addLayout(sel_row)
        top.addWidget(side)
        top.setStretchFactor(0, 4)
        top.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)
        splitter.setSizes([440, 200])
        outer.addWidget(splitter, 1)

    def _build_study_box(self) -> QGroupBox:
        """Komfortgewicht-Studie: beliebig viele w-Werte eingeben, trainieren, automatisch vergleichen."""
        box = QGroupBox('🧪 Komfortgewicht-Studie — mehrere w trainieren und zusammen vergleichen')
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel('w-Werte:'))
        self.study_weights = QLineEdit(DEFAULT_STUDY_WEIGHTS)
        self.study_weights.setPlaceholderText('beliebig viele, z. B. 2, 1, 0.5, 0.3, 0.1, 0.03')
        self.study_weights.setToolTip('So viele w-Werte wie gewünscht, getrennt durch Komma oder Semikolon. '
                                      'Dezimalkomma geht auch: 1,0; 0,3; 0,1')
        self.study_weights.textChanged.connect(self._update_study_preview)
        row.addWidget(self.study_weights, 1)
        row.addWidget(QLabel('Algorithmen:'))
        self.study_algos = {a: QCheckBox(a) for a in ALGOS}
        self.study_algos['SAC'].setChecked(True)
        for cb in self.study_algos.values():
            cb.toggled.connect(self._update_study_preview)
            row.addWidget(cb)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        self.study_skip = QCheckBox('Bereits trainierte w wiederverwenden')
        self.study_skip.setChecked(True)
        self.study_skip.setToolTip('Gibt es für ein w schon ein Modell (z. B. models/sac_seed0_w0_3.zip), '
                                   'wird es nicht neu trainiert, sondern direkt verglichen.')
        self.study_skip.toggled.connect(self._update_study_preview)
        self.study_btn = QPushButton('▶ Trainieren && vergleichen')
        self.study_btn.setProperty('accent', True)
        self.study_btn.setEnabled(self.start_study is not None)
        self.study_btn.clicked.connect(self._on_start_study)
        row2.addWidget(self.study_skip)
        row2.addStretch(1)
        row2.addWidget(self.study_btn)
        layout.addLayout(row2)

        self.study_info = QLabel()
        self.study_info.setWordWrap(True)
        self.study_info.setStyleSheet(f'color: {MUTED};')
        layout.addWidget(self.study_info)
        self._update_study_preview()
        return box

    def _study_selection(self):
        algos = [a for a, cb in self.study_algos.items() if cb.isChecked()]
        return parse_weights(self.study_weights.text()), algos

    def _update_study_preview(self, *_):
        """Vorschau: wie viele Trainings entstehen, wie viele Modelle schon da sind, grobe Dauer."""
        try:
            weights, algos = self._study_selection()
        except ValueError as e:
            self.study_info.setText(f'⚠️ {e}')
            return
        if not algos:
            self.study_info.setText('⚠️ Mindestens einen Algorithmus wählen.')
            return
        existing = [(a, w) for a in algos for w in weights
                    if (self.models_dir / f'{run_tag(a, 0, variant_for_w(w))}.zip').exists()]
        n_total = len(algos) * len(weights)
        n_train = n_total - (len(existing) if self.study_skip.isChecked() else 0)
        ws = ', '.join(f'{w:g}' for w in weights)
        text = f'{len(weights)} w-Werte ({ws}) × {len(algos)} Algorithmus/-en = {n_total} Modelle'
        if self.study_skip.isChecked() and existing:
            text += f', davon {len(existing)} schon trainiert'
        text += (f' → {n_train} Trainings (höchstens 3 gleichzeitig, bei 20 000 Schritten je ~2.5–3 h). '
                 'Hyperparameter und übrige Belohnungswerte kommen aus den Formularen links. '
                 '(Vorschau für Seed 0; der Seed links gilt beim Start.)')
        self.study_info.setText(text)

    def _on_start_study(self):
        try:
            weights, algos = self._study_selection()
        except ValueError as e:
            QMessageBox.warning(self, 'w-Werte ungültig', str(e))
            return
        if not algos:
            QMessageBox.warning(self, 'Kein Algorithmus', 'Bitte mindestens einen Algorithmus wählen.')
            return
        result = self.start_study(weights, algos, self.study_skip.isChecked())
        started, reused = result['started'], result['reused']
        if not started:
            self.set_study_status(f'Alle {len(reused)} Modelle existieren schon — vergleiche direkt.')
            self.study_finished(reused)
        else:
            self.set_study_status(f'{len(started)} Trainings gestartet/eingereiht'
                                  + (f', {len(reused)} vorhandene Modelle wiederverwendet' if reused else '')
                                  + '. Danach wird automatisch verglichen.')

    def set_study_status(self, text: str):
        self.study_info.setText('🧪 ' + text)

    def study_finished(self, model_names: list[str]):
        """Vom Reiter "Agent" gerufen, wenn alle Trainings der Studie fertig sind: vergleichen
        und danach genau die Studien-Modelle (plus RBC) anzeigen."""
        self._pending_select = [RBC] + list(model_names)
        ok, reason = self.can_run()
        if not ok:
            self.set_study_status(f'Trainings fertig, Vergleich wartet: {reason}')
            return
        self.set_study_status('Trainings fertig — Vergleich läuft …')
        self._run(force=False)

    def select_only(self, names):
        wanted = set(names)
        self.model_list.blockSignals(True)
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            on = item.data(Qt.ItemDataRole.UserRole) in wanted
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.model_list.blockSignals(False)
        self._render()

    # ---------- Berechnen ----------

    def _load_cached(self):
        if self.cache_path.exists():
            try:
                self.df = pd.read_csv(self.cache_path)
            except (OSError, ValueError, pd.errors.EmptyDataError):
                self.df = pd.DataFrame()
            if not self.df.empty:
                self.state_label.setText('Zwischengespeicherter Stand — „▶ Vergleich berechnen“ '
                                         'wertet neue Modelle aus.')
        self._show()

    def mark_outdated(self):
        """Vom Trainings-Reiter gerufen, wenn neue/neu trainierte Modelle auftauchen."""
        if self.worker is None and not self.df.empty:
            self.state_label.setText('Neue oder neu trainierte Modelle vorhanden — '
                                     '„▶ Vergleich berechnen“ aktualisiert.')

    def _run(self, force: bool):
        ok, reason = self.can_run()
        if not ok:
            QMessageBox.information(self, 'Gerade nicht möglich', reason)
            return
        self.run_btn.setEnabled(False)
        self.force_btn.setEnabled(False)
        self.state_label.setText('Starte …')
        self.worker = CompareWorker(self.models_dir, self.cache_path, force)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, i, n, name):
        if name:
            self.state_label.setText(f'Simuliere Testperiode {i + 1}/{n}: {name} … '
                                     '(je Modell ca. 0.5–1 min)')

    def _on_done(self, df):
        self.df = df
        self.worker = None
        self.run_btn.setEnabled(True)
        self.force_btn.setEnabled(True)
        n_err = int(df['error'].notna().sum()) if 'error' in df else 0
        self.state_label.setText('Fertig.' + (f' {n_err} Modell(e) nicht auswertbar, siehe Spalte „Hinweis“.'
                                              if n_err else ''))
        self._show()
        if self._pending_select:
            self.select_only(self._pending_select)
            n = len(self._pending_select) - 1
            self.set_study_status(f'Studie ausgewertet: {n} Modelle + RBC ausgewählt '
                                  '(weitere Modelle in der Liste rechts zuschaltbar).')
            self._pending_select = None

    def _on_failed(self, msg):
        self.worker = None
        self.run_btn.setEnabled(True)
        self.force_btn.setEnabled(True)
        self.state_label.setText('Fehlgeschlagen.')
        QMessageBox.warning(self, 'Vergleich fehlgeschlagen',
                            f'{msg}\n\nLäuft BOPTEST? (scripts\\start_boptest.ps1)')

    # ---------- Auswahl (Achsen, Referenz, Modelle) ----------

    def _valid(self) -> pd.DataFrame:
        df = self.df
        if df.empty or 'cost_eur' not in df:
            return df
        return df[df['error'].isna() & df['cost_eur'].notna()]

    def _available_metrics(self) -> list[str]:
        """Nur Kennzahlen mit mindestens einem Wert (z. B. keine Gas-Spitzenlast bei einer
        Wärmepumpe); w nur, wenn es Studien-Modelle gibt."""
        valid = self._valid()
        keys = []
        for key in METRICS:
            if key == 'savings_pct' or (key in valid and valid[key].notna().any()):
                keys.append(key)
        return keys

    @staticmethod
    def _fill_combo(combo: QComboBox, items: list[tuple[str, str]], keep: str | None, default: str):
        """Einträge (Anzeigetext, Schlüssel) setzen und die bisherige Auswahl behalten."""
        combo.blockSignals(True)
        combo.clear()
        for text, key in items:
            combo.addItem(text, key)
        keys = [k for _, k in items]
        target = keep if keep in keys else default if default in keys else (keys[0] if keys else None)
        if target is not None:
            combo.setCurrentIndex(keys.index(target))
        combo.blockSignals(False)

    def _refresh_choices(self):
        metrics = self._available_metrics()
        items = [(METRICS[k][0], k) for k in metrics]
        self._fill_combo(self.x_combo, items, self._choice['x'], DEFAULT_X)
        self._fill_combo(self.y_combo, items, self._choice['y'], DEFAULT_Y)

        valid = self._valid()
        refs = [(display_name(r), r['name']) for _, r in valid.iterrows()] if not valid.empty else []
        self._fill_combo(self.ref_combo, refs, self._choice['ref'], RBC)

        # Modellliste: neue Modelle angehakt, bisherige Haken bleiben erhalten
        previous = {self.model_list.item(i).data(Qt.ItemDataRole.UserRole): self.model_list.item(i).checkState()
                    for i in range(self.model_list.count())}
        self.model_list.blockSignals(True)
        self.model_list.clear()
        for _, r in (valid.iterrows() if not valid.empty else []):
            item = QListWidgetItem(display_name(r))
            item.setData(Qt.ItemDataRole.UserRole, r['name'])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(previous.get(r['name'], Qt.CheckState.Checked))
            self.model_list.addItem(item)
        self.model_list.blockSignals(False)

    def _checked_names(self) -> list[str]:
        return [self.model_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.model_list.count())
                if self.model_list.item(i).checkState() == Qt.CheckState.Checked]

    def _check_all(self, on: bool):
        self.model_list.blockSignals(True)
        for i in range(self.model_list.count()):
            self.model_list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.model_list.blockSignals(False)
        self._render()

    def _swap_axes(self):
        x, y = self.x_combo.currentIndex(), self.y_combo.currentIndex()
        self.x_combo.blockSignals(True)
        self.x_combo.setCurrentIndex(y)
        self.x_combo.blockSignals(False)
        self.y_combo.setCurrentIndex(x)   # löst das Neuzeichnen aus (und merkt beide Achsen)

    def _on_choice_changed(self, _index=None):
        # nur echte Nutzerauswahl (beim Befüllen sind die Signale blockiert)
        for key, combo in (('x', self.x_combo), ('y', self.y_combo), ('ref', self.ref_combo)):
            if combo.currentData() is not None:
                self._choice[key] = combo.currentData()
        self._render()

    # ---------- Anzeige ----------

    def _show(self):
        self._refresh_choices()
        self._render()

    def _view(self) -> pd.DataFrame:
        """Gültige Zeilen, Ersparnis gegen die gewählte Referenz, nur angehakte Modelle."""
        valid = self._valid()
        if valid.empty:
            return valid
        view = valid.copy()
        view['savings_pct'] = savings_vs(view, self.ref_combo.currentData() or RBC)
        return view[view.name.isin(self._checked_names())]

    def _render(self):
        reference = self.ref_combo.currentData() or RBC
        lines = summary_sentences(self.df, reference, names=self._checked_names()) if not self.df.empty else []
        self.summary.setText('\n'.join(lines) if lines else 'Noch kein Vergleich berechnet.')
        self.copy_btn.setEnabled(bool(lines))
        self._fill_table()
        self._draw()

    def _copy_summary(self):
        QApplication.clipboard().setText(self.summary.text())

    def _fill_table(self):
        view = self._view()
        errors = self.df[self.df['error'].notna()] if 'error' in self.df else pd.DataFrame()
        metrics = self._available_metrics()
        headers = ['Modell'] + [METRICS[k][0] for k in metrics] + ['Hinweis']
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        rows = list(view.iterrows()) + list(errors.iterrows())
        self.table.setRowCount(len(rows))
        reference = self.ref_combo.currentData() or RBC
        for r, (_, row) in enumerate(rows):
            name = display_name(row) + ('  (Referenz)' if row['name'] == reference else '')
            self.table.setItem(r, 0, QTableWidgetItem(name))
            for c, key in enumerate(metrics, start=1):
                v = row.get(key)
                text = '' if key == 'savings_pct' and row['name'] == reference else fmt(key, v)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
            err = row.get('error')
            self.table.setItem(r, len(headers) - 1, QTableWidgetItem('' if pd.isna(err) else str(err)))

    def _draw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._points = []
        self._ax = self._hover = None
        view = self._view()
        xk, yk = self.x_combo.currentData(), self.y_combo.currentData()
        if view.empty or not xk or not yk:
            msg = ('Diagramm erscheint hier nach „Vergleich berechnen“ oder einer Studie.' if self._valid().empty
                   else 'Kein Modell ausgewählt.')
            ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes, fontsize=9, color=MUTED)
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            return
        view = view[view[xk].notna() & view[yk].notna()]
        reference = self.ref_combo.currentData() or RBC

        for algo, group in view[view.name != RBC].groupby('algo'):
            color, marker = ALGO_STYLE.get(algo, ('#B5CEA8', 'D'))
            g = group.sort_values('w', ascending=False, na_position='last')
            with_w = g[g.w.notna()]
            if len(with_w) > 1:   # die Kurve: Varianten desselben Algorithmus nach w verbinden
                ax.plot(with_w[xk], with_w[yk], color=color, lw=2, zorder=2)
            ax.scatter(g[xk], g[yk], s=110, color=color, marker=marker, zorder=3,
                       edgecolors='#1e1e1e', linewidths=1.5, label=algo)
            dx, dy = LABEL_OFFSET.get(algo, (8, 7))
            for _, r in g.iterrows():
                tag = f'w={r.w:g}' if pd.notna(r.w) else r['name']
                ax.annotate(tag, (r[xk], r[yk]), xytext=(dx, dy), textcoords='offset points',
                            fontsize=8, color=TEXT, ha='left' if dx > 0 else 'right', va='center')
                self._points.append((r[xk], r[yk], self._tooltip(r, xk, yk, reference)))
        rbc = view[view.name == RBC]
        if not rbc.empty:
            r = rbc.iloc[0]
            color, marker = RBC_STYLE
            # Direkt beschriftet (Annotation daneben) — deshalb kein Legendeneintrag.
            ax.scatter([r[xk]], [r[yk]], s=280, color=color, marker=marker, zorder=4,
                       edgecolors='#1e1e1e', linewidths=1.5)
            ax.annotate('RBC', (r[xk], r[yk]), xytext=(-10, 0), textcoords='offset points',
                        fontsize=8, color=TEXT, ha='right', va='center')
            self._points.append((r[xk], r[yk], self._tooltip(r, xk, yk, reference)))

        # Referenz: Ring um den Punkt + gestrichelte Linien durch ihn (teilen in besser/schlechter)
        ref = view[view.name == reference]
        if not ref.empty:
            rx, ry = ref.iloc[0][xk], ref.iloc[0][yk]
            ax.scatter([rx], [ry], s=520, facecolors='none', edgecolors=TEXT, linewidths=1.2, zorder=5)
            ax.axhline(ry, color=MUTED, lw=1, ls='--', alpha=0.6, zorder=1)
            ax.axvline(rx, color=MUTED, lw=1, ls='--', alpha=0.6, zorder=1)

        ax.set_xlabel(METRICS[xk][0])
        ax.set_ylabel(METRICS[yk][0])
        ax.set_title(self._better_hint(xk, yk), loc='left', fontsize=8, color=MUTED)
        if len(view) == 1:
            # Nur ein Punkt (z. B. noch kein Modell trainiert): matplotlib würde auf einen winzigen
            # Bereich um ihn zoomen — stattdessen 0 bis doppelter Wert, Punkt in der Mitte.
            r = view.iloc[0]
            for v, set_ in ((r[xk], ax.set_xlim), (r[yk], ax.set_ylim)):
                set_((0, 2 * v) if v > 0 else (2 * v, 0) if v < 0 else (-1, 1))
            ax.text(0.5, 0.04, 'Noch keine trainierten Modelle — sie erscheinen hier nach dem Training '
                    'und „Vergleich berechnen“.', transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=8, color=MUTED)
        else:
            for get, set_, pad in ((ax.get_xlim, ax.set_xlim, 0.08), (ax.get_ylim, ax.set_ylim, 0.06)):
                lo, hi = get()   # Platz für die Labels an allen Rändern
                set_(lo - pad * (hi - lo), hi + pad * (hi - lo))
        if ax.get_legend_handles_labels()[0]:   # nur Algorithmen; RBC ist direkt beschriftet
            ax.legend(loc='best')
        self._hover = ax.annotate('', (0, 0), xytext=(12, 12), textcoords='offset points', fontsize=8,
                                  color=TEXT, bbox=dict(boxstyle='round,pad=0.4', fc='#252526', ec='#3c3c3c'),
                                  zorder=10)
        self._hover.set_visible(False)
        self._ax = ax
        self.canvas.draw_idle()

    @staticmethod
    def _better_hint(xk: str, yk: str) -> str:
        """'Besser = unten links · Ring + gestrichelte Linien = Referenz' — aus der Richtung beider Kennzahlen."""
        vertical = {'low': 'unten', 'high': 'oben'}.get(METRICS[yk][1])
        horizontal = {'low': 'links', 'high': 'rechts'}.get(METRICS[xk][1])
        where = ' '.join(p for p in (vertical, horizontal) if p)
        return (f'Besser = {where} · ' if where else '') + 'Ring + gestrichelte Linien = Referenz'

    @staticmethod
    def _tooltip(r, xk: str, yk: str, reference: str) -> str:
        lines = [display_name(r) + ('  (Referenz)' if r['name'] == reference else '')]
        keys = list(dict.fromkeys([xk, yk, 'cost_eur', 'savings_pct', 'tdis_kh', 'grid_kwh']))
        for key in keys:
            if key == 'savings_pct' and r['name'] == reference:
                continue
            text = fmt(key, r.get(key))
            if text:
                lines.append(f'{METRICS[key][0]}: {text}')
        return '\n'.join(lines)

    def _on_hover(self, event):
        if self._ax is None or not self._points or event.inaxes is not self._ax:
            if self._hover is not None and self._hover.get_visible():
                self._hover.set_visible(False)
                self.canvas.draw_idle()
            return
        trans = self._ax.transData
        pix = np.array([trans.transform((x, y)) for x, y, _ in self._points])
        d = np.hypot(pix[:, 0] - event.x, pix[:, 1] - event.y)
        i = int(d.argmin())
        visible = d[i] < 14   # Trefferbereich größer als der Marker
        if visible:
            x, y, text = self._points[i]
            self._hover.xy = (x, y)
            self._hover.set_text(text)
        if visible or self._hover.get_visible():
            self._hover.set_visible(visible)
            self.canvas.draw_idle()
