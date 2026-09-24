"""Qt-Ansicht "Vergleich": Kosten-Komfort-Kurve über alle trainierten Modelle gegen den RBC.

Gedacht für die Komfortgewicht-Studie (Reiter "Training", Belohnungsformular): mehrere
Varianten mit unterschiedlichem w trainieren, hier auf derselben, nie im Training gesehenen
Testperiode auswerten und nebeneinanderstellen — nicht ein Ergebnis, sondern die ganze
Abwägung: „Bei w=1 kaum Ersparnis, bei w=0.1 sparen wir X %, aber Y Kelvin-Stunden
Komfortverletzung“.

Die Simulation läuft in einem Hintergrund-Thread (je Modell eine Testepisode, ~0.5-1 min),
Ergebnisse werden in results/compare.csv zwischengespeichert (logic/evaluation.py::compare).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from logic.evaluation import RBC, compare, summary_sentences

# Farbe UND Markerform je Algorithmus (Identität nie nur über Farbe), fest je Algorithmus —
# nicht nach Reihenfolge vergeben. Gleiche Farben wie im Reiter "Beobachten".
ALGO_STYLE = {'SAC': ('#4FC1FF', 'o'), 'PPO': ('#4EC9B0', 's'), 'TD3': ('#F48771', '^')}
# Beschriftungs-Versatz je Algorithmus (Punkte), damit sich die "w=…"-Labels verschiedener
# Algorithmen am selben Ort (typisch: alle w=1 nahe beim RBC) nicht überdecken.
LABEL_OFFSET = {'SAC': (8, -13), 'TD3': (8, 7), 'PPO': (-8, 7)}
RBC_STYLE = ('#DCDCAA', '*')
TEXT = '#cccccc'
MUTED = '#8A98A3'

COLUMNS = [('name', 'Modell'), ('algo', 'Algorithmus'), ('w', 'w'), ('cost_eur', 'Stromkosten (€)'),
           ('savings_pct', 'Ersparnis ggü. RBC'), ('tdis_kh', 'Komfortverletzung (K·h)'),
           ('grid_kwh', 'Netzbezug (kWh)'), ('error', 'Hinweis')]


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
    def __init__(self, models_dir: Path, cache_path: Path, can_run=None, parent=None):
        super().__init__(parent)
        self.models_dir = Path(models_dir)
        self.cache_path = Path(cache_path)
        self.can_run = can_run or (lambda: (True, ''))
        self.worker = None
        self.df = pd.DataFrame()
        self._points = []   # (x, y, Tooltip-Text) für den Hover
        self._build_ui()
        self._load_cached()

    # ---------- Aufbau ----------

    def _build_ui(self):
        outer = QVBoxLayout(self)

        intro = QLabel('Alle trainierten Modelle und der RBC auf derselben Testperiode. Für eine '
                       'Kosten-Komfort-Kurve links im Belohnungsformular die <b>Komfortgewicht-Studie</b> '
                       'aktivieren (z. B. w = 1.0, 0.3, 0.1), trainieren, dann hier berechnen.')
        intro.setWordWrap(True)
        outer.addWidget(intro)

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
        row.addWidget(self.run_btn)
        row.addWidget(self.force_btn)
        row.addWidget(self.state_label, 1)
        outer.addLayout(row)

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

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.figure = Figure(figsize=(7, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        splitter.addWidget(self.canvas)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)
        splitter.setSizes([420, 200])
        outer.addWidget(splitter, 1)

        self._draw()

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

    def _on_failed(self, msg):
        self.worker = None
        self.run_btn.setEnabled(True)
        self.force_btn.setEnabled(True)
        self.state_label.setText('Fehlgeschlagen.')
        QMessageBox.warning(self, 'Vergleich fehlgeschlagen',
                            f'{msg}\n\nLäuft BOPTEST? (scripts\\start_boptest.ps1)')

    # ---------- Anzeige ----------

    def _show(self):
        lines = summary_sentences(self.df) if not self.df.empty else []
        self.summary.setText('\n'.join(lines) if lines else
                             'Noch kein Vergleich berechnet.')
        self.copy_btn.setEnabled(bool(lines))
        self._fill_table()
        self._draw()

    def _copy_summary(self):
        QApplication.clipboard().setText(self.summary.text())

    def _fill_table(self):
        df = self.df
        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, (key, _) in enumerate(COLUMNS):
                v = row.get(key)
                if key == 'error':
                    text = '' if pd.isna(v) else str(v)
                elif key == 'w':
                    text = '' if pd.isna(v) else f'{v:g}'
                elif key == 'savings_pct':
                    text = '' if pd.isna(v) or row.get('name') == RBC else f'{v:+.1f} %'
                elif isinstance(v, float):
                    text = '' if pd.isna(v) else f'{v:.2f}'
                else:
                    text = '' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
                item = QTableWidgetItem(text)
                if key not in ('name', 'algo', 'error'):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)

    def _draw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._points = []
        df = self.df
        ok = df[df['error'].isna() & df['cost_eur'].notna()] if not df.empty else df
        if ok is None or ok.empty:
            ax.text(0.5, 0.5, 'Kosten-Komfort-Kurve erscheint hier nach „▶ Vergleich berechnen“.',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9, color=MUTED)
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            return

        rbc = ok[ok.name == RBC]
        for algo, group in ok[ok.name != RBC].groupby('algo'):
            color, marker = ALGO_STYLE.get(algo, ('#B5CEA8', 'D'))
            g = group.sort_values('w', ascending=False, na_position='last')
            with_w = g[g.w.notna()]
            if len(with_w) > 1:   # die Kurve: Varianten desselben Algorithmus nach w verbinden
                ax.plot(with_w.tdis_kh, with_w.cost_eur, color=color, lw=2, zorder=2)
            ax.scatter(g.tdis_kh, g.cost_eur, s=110, color=color, marker=marker, zorder=3,
                       edgecolors='#1e1e1e', linewidths=1.5, label=algo)
            dx, dy = LABEL_OFFSET.get(algo, (8, 7))
            for _, r in g.iterrows():
                tag = f'w={r.w:g}' if pd.notna(r.w) else r['name']
                ax.annotate(tag, (r.tdis_kh, r.cost_eur), xytext=(dx, dy), textcoords='offset points',
                            fontsize=8, color=TEXT, ha='left' if dx > 0 else 'right', va='center')
                self._points.append((r.tdis_kh, r.cost_eur, self._tooltip(r)))
        if not rbc.empty:
            r = rbc.iloc[0]
            color, marker = RBC_STYLE
            ax.scatter([r.tdis_kh], [r.cost_eur], s=280, color=color, marker=marker, zorder=4,
                       edgecolors='#1e1e1e', linewidths=1.5, label='RBC (Referenz)')
            # gestrichelte Bezugslinie: alles darunter ist günstiger als der RBC
            ax.axhline(r.cost_eur, color=color, lw=1, ls='--', alpha=0.5, zorder=1)
            ax.annotate('RBC', (r.tdis_kh, r.cost_eur), xytext=(-10, 0), textcoords='offset points',
                        fontsize=8, color=TEXT, ha='right', va='center')
            self._points.append((r.tdis_kh, r.cost_eur, self._tooltip(r)))

        ax.set_xlabel('Komfortverletzung im Testzeitraum (K·h)  →  schlechter')
        ax.set_ylabel('Stromkosten im Testzeitraum (€)  →  teurer')
        ax.set_title('Besser = unten links (günstiger und komfortabler) · gestrichelt = Kosten des RBC',
                     loc='left', fontsize=8, color=MUTED)
        lo, hi = ax.get_xlim()   # Platz für die Labels an allen Rändern
        ax.set_xlim(min(-0.08 * (hi - lo), lo), hi + 0.08 * (hi - lo))
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
        ax.legend(loc='upper right')
        self._hover = ax.annotate('', (0, 0), xytext=(12, 12), textcoords='offset points', fontsize=8,
                                  color=TEXT, bbox=dict(boxstyle='round,pad=0.4', fc='#252526', ec='#3c3c3c'),
                                  zorder=10)
        self._hover.set_visible(False)
        self._ax = ax
        self.canvas.draw_idle()

    @staticmethod
    def _tooltip(r) -> str:
        head = 'RBC (Referenz)' if r['name'] == RBC else f"{r['name']}" + (f'  (w={r.w:g})' if pd.notna(r.w) else '')
        lines = [head, f'Stromkosten: {r.cost_eur:.2f} €']
        if r['name'] != RBC and pd.notna(r.get('savings_pct')):
            lines.append(f'Ersparnis ggü. RBC: {r.savings_pct:+.1f} %')
        lines += [f'Komfortverletzung: {r.tdis_kh:.1f} K·h', f'Netzbezug: {r.grid_kwh:.1f} kWh']
        return '\n'.join(lines)

    def _on_hover(self, event):
        if not self._points or event.inaxes is not getattr(self, '_ax', None):
            if getattr(self, '_hover', None) is not None and self._hover.get_visible():
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
