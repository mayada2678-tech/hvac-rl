"""Qt-Ansicht "Datensatz": einen BOPTEST-Testfall erkunden, bevor/ohne dass man trainiert —
Wetter, Strompreis(e), interne Lasten, Komfort-Sollwerte, CO2-Intensität, übers ganze Jahr,
plus eine vollständige Merkmalsliste und einen Datenqualitätsbericht. Zwischen allen bei
BOPTEST bereitgestellten Testfällen wechselbar (Auswahlfeld oben) — die Analyse passt sich
dynamisch an, was der jeweilige Testfall tatsächlich anbietet (siehe logic/dataset.py).
Die eigentliche Logik steckt in logic/dataset.py, hier nur Darstellung.
"""
import pandas as pd
import requests
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QSlider, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from gui import theme
from logic.dataset import (KNOWN_LABELS, PRICE_COLORS, TESTCASE, daily_profile, label,
                           list_all_points, list_testcases, load_year, price_points, quality_report,
                           summary)

FALLBACK_PALETTE = ['#4FC1FF', '#4EC9B0', '#DCDCAA', '#F48771', '#9CDCFE', '#C586C0', '#B5CEA8', '#D7BA7D']


class DatasetView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = None
        self._measurements = self._forecasts = self._inputs = {}
        self._build_ui()
        self._load_testcase_list()
        self.reload()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel('<b>Testfall:</b>'))
        self.testcase_combo = QComboBox()
        self.testcase_combo.setEditable(True)   # auch ein selbst hinzugefügter Testfall geht
        self.testcase_combo.setToolTip('Zwischen allen bei diesem BOPTEST-Dienst bereitgestellten '
                                       'Testfällen wechseln, oder den Namen eines selbst per BOPTEST '
                                       '„provision" hochgeladenen Testfalls eintippen (siehe workbench.md).')
        top.addWidget(self.testcase_combo, 1)
        self.reload_btn = QPushButton('🔄 Laden / neu laden')
        self.reload_btn.setProperty('accent', True)
        self.reload_btn.clicked.connect(self.reload)
        top.addWidget(self.reload_btn)
        layout.addLayout(top)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_overview_tab(), '🧾 Übersicht & Datenqualität')
        self.tabs.addTab(self._build_features_tab(), '🔬 Alle Merkmale')
        self.tabs.addTab(self._build_price_tab(), '💶 Preisszenarien')
        self.tabs.addTab(self._build_series_tab(), '📈 Zeitreihen')
        self.tabs.addTab(self._build_dist_tab(), '📊 Verteilung & Tagesprofil')

    def _load_testcase_list(self):
        try:
            names = list_testcases()
        except requests.exceptions.RequestException:
            names = [TESTCASE]
        self.testcase_combo.blockSignals(True)
        self.testcase_combo.addItems(names)
        self.testcase_combo.setCurrentText(TESTCASE if TESTCASE in names else (names[0] if names else TESTCASE))
        self.testcase_combo.blockSignals(False)

    def _current_testcase(self) -> str:
        return self.testcase_combo.currentText().strip() or TESTCASE

    def reload(self):
        testcase = self._current_testcase()
        self.error_label.setText('')
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            self._df = load_year(testcase)
            self._measurements, self._forecasts, self._inputs = list_all_points(testcase)
        except requests.exceptions.RequestException:
            self._df = None
            self._measurements = self._forecasts = self._inputs = {}
            self.error_label.setText('⚠️ BOPTEST ist unter http://127.0.0.1:8000 nicht erreichbar. '
                                     'Erst scripts\\start_boptest.ps1 ausführen (siehe README.md), '
                                     'dann "🔄 Laden" drücken.')
        except Exception as e:
            self._df = None
            self._measurements = self._forecasts = self._inputs = {}
            self.error_label.setText(f'⚠️ Testfall "{testcase}" konnte nicht geladen werden: {e}')
        finally:
            QApplication.restoreOverrideCursor()

        self._refresh_overview()
        self._refresh_features_tab()
        self._refresh_price_tab()
        self._refresh_series_controls()
        self._refresh_dist_controls()

    def _value_columns(self) -> list[str]:
        if self._df is None:
            return []
        return [c for c in self._df.columns if c not in ('time', 'day', 'hour')]

    def _color_for(self, point: str, ordered: list[str]) -> str:
        if point in PRICE_COLORS:
            return PRICE_COLORS[point]
        others = [p for p in ordered if p not in PRICE_COLORS]
        return FALLBACK_PALETTE[others.index(point) % len(FALLBACK_PALETTE)] if point in others else FALLBACK_PALETTE[0]

    # ---------- Übersicht ----------

    def _build_overview_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel('<b>Datensatz auf einen Blick</b> — Testfall, Merkmale, Zeitraum, Datenqualität'))
        self.stat_cards_layout = QGridLayout()
        self.stat_cards_layout.setSpacing(10)
        layout.addLayout(self.stat_cards_layout)

        layout.addWidget(QLabel('<b>Datenqualität</b> (ganzes simuliertes Jahr, stündlich):'))
        self.quality_cards_layout = QGridLayout()
        self.quality_cards_layout.setSpacing(10)
        layout.addLayout(self.quality_cards_layout)
        self.quality_note = QLabel()
        self.quality_note.setWordWrap(True)
        layout.addWidget(self.quality_note)
        layout.addStretch()
        return w

    def _make_stat_card(self, title: str, value: str) -> QFrame:
        frame = QFrame()
        frame.setProperty('card', True)
        v = QVBoxLayout(frame)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet(f'color: {theme.TEXT_MUTED}; font-size: 9pt; border: none;')
        val = QLabel(value)
        val.setStyleSheet('font-size: 14pt; font-weight: 600; border: none;')
        val.setWordWrap(True)
        v.addWidget(t)
        v.addWidget(val)
        return frame

    def _clear_grid(self, grid_layout: QGridLayout):
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _refresh_overview(self):
        self._clear_grid(self.stat_cards_layout)
        self._clear_grid(self.quality_cards_layout)
        self.quality_note.setText('')
        if self._df is None:
            return

        n_meas, n_fc, n_inp = len(self._measurements), len(self._forecasts), len(self._inputs)
        s = summary(self._df)
        cards = [
            ('Testfall', self._current_testcase()),
            ('Merkmale gesamt', str(n_meas + n_fc + n_inp)),
            ('davon Messgrößen', str(n_meas)),
            ('davon Vorhersagegrößen', str(n_fc)),
            ('davon Eingänge (Aktionen)', str(n_inp)),
            ('Stunden im Jahr', str(s['Stunden'])),
        ]
        for key, value in s.items():
            if key in ('Stunden', 'Tage'):
                continue
            lo, hi = value
            cards.append((key.replace('-Spanne', ''), f'{lo:.3f} – {hi:.3f}'))
        for i, (title, value) in enumerate(cards):
            self.stat_cards_layout.addWidget(self._make_stat_card(title, value), i // 5, i % 5)

        q = quality_report(self._df)
        quality_cards = [
            ('Vollständigkeit', f"{q['Vollständigkeit (%)']:.1f} %"),
            ('Geprüfte Spalten', str(q['Spalten geprüft'])),
            ('Fehlende Zellen', f"{q['davon fehlend']} / {q['Zellen gesamt']}"),
            ('Zeitlücken', str(q['Zeitlücken'])),
            ('Zeitraster', f"{q['Erwartetes Intervall (s)']} s"),
        ]
        for i, (title, value) in enumerate(quality_cards):
            self.quality_cards_layout.addWidget(self._make_stat_card(title, value), i // 5, i % 5)

        notes = []
        if q['Zeitlücken']:
            notes.append(f"⚠️ {q['Zeitlücken']} Zeitlücke(n) größer als erwartet gefunden.")
        if q['Konstante Spalten']:
            shown = ', '.join(q['Konstante Spalten'][:8]) + (' …' if len(q['Konstante Spalten']) > 8 else '')
            notes.append(f"ℹ️ {len(q['Konstante Spalten'])} konstante (nicht-variierende) Spalten: {shown} — "
                        'inhaltlich unauffällig, keine fehlerhaften Daten (z. B. ist ein konstantes '
                        'Preisszenario per Definition konstant).')
        if not notes:
            notes.append('✅ Keine fehlenden Werte, keine Zeitlücken.')
        self.quality_note.setText('  '.join(notes))

    # ---------- Alle Merkmale ----------

    def _build_features_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.features_summary_label = QLabel()
        layout.addWidget(self.features_summary_label)

        sub_tabs = QTabWidget()
        layout.addWidget(sub_tabs, 1)
        self.meas_table = QTableWidget()
        sub_tabs.addTab(self.meas_table, 'Messgrößen')
        self.forecast_table = QTableWidget()
        sub_tabs.addTab(self.forecast_table, 'Vorhersagegrößen (Randbedingungen)')
        self.inputs_table = QTableWidget()
        sub_tabs.addTab(self.inputs_table, 'Eingänge (Stellgrößen)')
        return w

    def _fill_meta_table(self, table: QTableWidget, points: dict):
        cols = ['Name', 'Einheit', 'Minimum', 'Maximum', 'Beschreibung']
        table.setRowCount(len(points))
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        for i, (name, info) in enumerate(sorted(points.items())):
            values = [name, info.get('Unit') or '—',
                     '—' if info.get('Minimum') is None else str(info['Minimum']),
                     '—' if info.get('Maximum') is None else str(info['Maximum']),
                     info.get('Description') or '—']
            for j, v in enumerate(values):
                item = QTableWidgetItem(v)
                if j == 0:
                    item.setForeground(QColor(theme.PLOT_PALETTE[0]))
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    def _refresh_features_tab(self):
        n_meas, n_fc, n_inp = len(self._measurements), len(self._forecasts), len(self._inputs)
        self.features_summary_label.setText(
            f'<b>{n_meas + n_fc + n_inp} Merkmale insgesamt</b> in „{self._current_testcase()}" — '
            f'{n_meas} Messgrößen (was die Simulation ausgibt), {n_fc} Vorhersagegrößen '
            f'(Wetter/Preis/Lasten als Randbedingung), {n_inp} Eingänge (was der Agent bzw. der '
            'Regler steuern kann).')
        self._fill_meta_table(self.meas_table, self._measurements)
        self._fill_meta_table(self.forecast_table, self._forecasts)
        self._fill_meta_table(self.inputs_table, self._inputs)

    # ---------- Preisszenarien ----------

    def _build_price_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.price_note = QLabel('Alle Preis-Randbedingungen dieses Testfalls im Vergleich — welche beim '
                                 'Training gewählt wird, bestimmt `scenario` in logic/envs.py::make_env().')
        self.price_note.setWordWrap(True)
        layout.addWidget(self.price_note)
        win = QHBoxLayout()
        win.addWidget(QLabel('von Tag:'))
        self.price_lo_slider = QSlider(Qt.Horizontal)
        self.price_lo_slider.valueChanged.connect(self._redraw_price)
        win.addWidget(self.price_lo_slider)
        win.addWidget(QLabel('bis Tag:'))
        self.price_hi_slider = QSlider(Qt.Horizontal)
        self.price_hi_slider.valueChanged.connect(self._redraw_price)
        win.addWidget(self.price_hi_slider)
        layout.addLayout(win)
        self.price_figure = Figure(figsize=(9, 4), constrained_layout=True)
        self.price_canvas = FigureCanvas(self.price_figure)
        layout.addWidget(self.price_canvas, 1)
        return w

    def _refresh_price_tab(self):
        if self._df is None:
            return
        n_days = int(self._df['day'].max()) + 1
        for slider, default in ((self.price_lo_slider, 0), (self.price_hi_slider, min(7, n_days))):
            slider.blockSignals(True)
            slider.setRange(0, n_days)
            slider.setValue(default)
            slider.blockSignals(False)
        self._redraw_price()

    def _redraw_price(self):
        if self._df is None:
            return
        cols = [c for c in price_points(dict.fromkeys(self._df.columns)) if c in self._df.columns]
        lo, hi = sorted((self.price_lo_slider.value(), self.price_hi_slider.value()))
        if hi <= lo:
            hi = lo + 1
        d = self._df[(self._df.day >= lo) & (self._df.day < hi)]
        self.price_figure.clear()
        ax = self.price_figure.add_subplot(111)
        if not cols:
            ax.text(0.5, 0.5, 'Dieser Testfall bietet keine Preis-Randbedingung an.', ha='center', va='center',
                   transform=ax.transAxes, color=theme.TEXT_MUTED)
        for col in cols:
            ax.plot(d.time / 3600, d[col], color=self._color_for(col, cols), label=label(col), lw=1.3)
        ax.set_xlabel('Stunde des Jahres')
        ax.set_ylabel('Preis (pro kWh)')
        if cols:
            ax.legend(fontsize=8)
        self.price_canvas.draw_idle()

    # ---------- Zeitreihen ----------

    def _build_series_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel('Größe:'))
        self.series_combo = QComboBox()
        self.series_combo.currentIndexChanged.connect(self._redraw_series)
        top.addWidget(self.series_combo, 1)
        layout.addLayout(top)

        win = QHBoxLayout()
        win.addWidget(QLabel('von Tag:'))
        self.series_lo_slider = QSlider(Qt.Horizontal)
        self.series_lo_slider.valueChanged.connect(self._redraw_series)
        win.addWidget(self.series_lo_slider)
        win.addWidget(QLabel('bis Tag:'))
        self.series_hi_slider = QSlider(Qt.Horizontal)
        self.series_hi_slider.valueChanged.connect(self._redraw_series)
        win.addWidget(self.series_hi_slider)
        layout.addLayout(win)

        self.series_figure = Figure(figsize=(9, 4), constrained_layout=True)
        self.series_canvas = FigureCanvas(self.series_figure)
        layout.addWidget(self.series_canvas, 1)
        return w

    def _populate_feature_combo(self, combo: QComboBox):
        """Combo-Inhalt neu befüllen, ohne die (einmalig beim Aufbau verbundene) Signal-
        Verbindung anzufassen — `blockSignals` verhindert währenddessen unnötige Auslöser."""
        combo.blockSignals(True)
        combo.clear()
        for col in self._value_columns():
            combo.addItem(label(col), col)
        combo.blockSignals(False)

    def _refresh_series_controls(self):
        self._populate_feature_combo(self.series_combo)
        if self._df is None:
            return
        n_days = int(self._df['day'].max()) + 1
        for slider, default in ((self.series_lo_slider, 0), (self.series_hi_slider, min(14, n_days))):
            slider.blockSignals(True)
            slider.setRange(0, n_days)
            slider.setValue(default)
            slider.blockSignals(False)
        self._redraw_series()

    def _redraw_series(self):
        if self._df is None:
            return
        col = self.series_combo.currentData()
        lo, hi = sorted((self.series_lo_slider.value(), self.series_hi_slider.value()))
        if hi <= lo:
            hi = lo + 1
        d = self._df[(self._df.day >= lo) & (self._df.day < hi)]
        self.series_figure.clear()
        ax = self.series_figure.add_subplot(111)
        if col in d.columns:
            ax.plot(d.time / 3600, d[col], color=theme.PLOT_PALETTE[0], lw=1.3)
        ax.set_xlabel('Stunde des Jahres')
        ax.set_ylabel(label(col) if col else '')
        self.series_canvas.draw_idle()

    # ---------- Verteilung & Tagesprofil ----------

    def _build_dist_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel('Größe:'))
        self.dist_combo = QComboBox()
        self.dist_combo.currentIndexChanged.connect(self._redraw_dist)
        top.addWidget(self.dist_combo, 1)
        layout.addLayout(top)

        layout.addWidget(QLabel('Verteilung übers Jahr:'))
        self.dist_hist_figure = Figure(figsize=(9, 2.5), constrained_layout=True)
        self.dist_hist_canvas = FigureCanvas(self.dist_hist_figure)
        layout.addWidget(self.dist_hist_canvas)

        layout.addWidget(QLabel('Mittlerer Tagesverlauf:'))
        self.dist_profile_figure = Figure(figsize=(9, 2.5), constrained_layout=True)
        self.dist_profile_canvas = FigureCanvas(self.dist_profile_figure)
        layout.addWidget(self.dist_profile_canvas)
        return w

    def _refresh_dist_controls(self):
        self._populate_feature_combo(self.dist_combo)
        self._redraw_dist()

    def _redraw_dist(self):
        if self._df is None:
            return
        col = self.dist_combo.currentData()
        if not col or col not in self._df.columns:
            self.dist_hist_figure.clear()
            self.dist_hist_canvas.draw_idle()
            self.dist_profile_figure.clear()
            self.dist_profile_canvas.draw_idle()
            return

        self.dist_hist_figure.clear()
        ax = self.dist_hist_figure.add_subplot(111)
        ax.hist(self._df[col].dropna(), bins=40, color=theme.PLOT_PALETTE[0])
        ax.set_xlabel(label(col))
        ax.set_ylabel('Stunden')
        self.dist_hist_canvas.draw_idle()

        self.dist_profile_figure.clear()
        ax = self.dist_profile_figure.add_subplot(111)
        prof = daily_profile(self._df, col)
        ax.plot(prof.index, prof.values, color=theme.PLOT_PALETTE[1], marker='o', markersize=3)
        ax.set_xlabel('Stunde des Tages')
        ax.set_ylabel(f'Ø {label(col)}')
        self.dist_profile_canvas.draw_idle()
