"""Qt-Baustein "Beobachten": Stunde-für-Stunde-Animation der Testperiode (3 Tage im Februar,
siehe logic/envs.py) — Zonentemperatur vs. Komfortband, Wärmepumpen-Modulation, dynamischer
Strompreis. Strategien: BOPTESTs eingebauter Regler (RBC) und beliebig viele trainierte
Modelle, zum direkten Vergleich.
"""
from pathlib import Path

import pandas as pd
import requests
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)
from stable_baselines3 import PPO, SAC, TD3

from logic.watch import record

ALGOS = {'sac': SAC, 'td3': TD3, 'ppo': PPO}
# Feste Farbzuordnung je Strategie (kategorial, nach Auswahlreihenfolge vergeben, nie nach
# Rang neu eingefärbt) — Solar/Preis bekommen eigene, hiervon garantiert verschiedene Farben,
# damit z. B. "Regel (RBC)" und "Solarerzeugung" nicht (wie zuvor) versehentlich dieselbe
# Farbe teilen und ununterscheidbar werden.
COLORS = {'Regel (RBC)': '#DCDCAA'}
AGENT_COLOR = '#4FC1FF'
AGENT_PALETTE = ['#4FC1FF', '#4EC9B0', '#F48771', '#C586C0', '#B5CEA8']
SOLAR_COLOR = '#D7BA7D'
PRICE_COLOR = '#9d9d9d'

KPI_LABELS = {
    'cost_tot': 'Kosten (€ bzw. $/m²)', 'tdis_tot': 'Komfort-Defizit (Kh/m²)',
    'idis_tot': 'Unbehaglichkeits-Index', 'ener_tot': 'Energie (kWh/m²)',
    'emis_tot': 'CO2-Emissionen (kg/m²)', 'time_rat': 'Rechenzeit-Verhältnis',
}


class WatchView(QWidget):
    """Strategien wählen (RBC + trainierte Modelle), die Testperiode abspielen oder per
    Regler durchblättern."""

    def __init__(self, models_dir: Path, parent=None):
        super().__init__(parent)
        self.models_dir = Path(models_dir)
        self._cache: dict[tuple[str, float], tuple[pd.DataFrame, dict]] = {}
        self._runs: dict[str, pd.DataFrame] = {}
        self._kpis: dict[str, dict] = {}
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.reload_models()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.strategy_list = QListWidget()
        self.strategy_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.strategy_list.setMaximumHeight(90)
        self.strategy_list.itemSelectionChanged.connect(self._on_selection_changed)
        top.addWidget(QLabel('Strategien:'))
        top.addWidget(self.strategy_list, 1)
        self.reload_btn = QPushButton('⟳ Modelle neu laden')
        self.reload_btn.clicked.connect(self.reload_models)
        top.addWidget(self.reload_btn)
        layout.addLayout(top)

        controls = QHBoxLayout()
        self.play_btn = QPushButton('▶ Abspielen')
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggled)
        self.hour_slider = QSlider(Qt.Horizontal)
        self.hour_slider.valueChanged.connect(self._redraw)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 1000)
        self.speed_slider.setValue(250)
        self.speed_slider.setMaximumWidth(120)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.hour_slider, 1)
        controls.addWidget(QLabel('Tempo:'))
        controls.addWidget(self.speed_slider)
        layout.addLayout(controls)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.figure = Figure(figsize=(10, 10), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

        layout.addWidget(QLabel('Kennzahlen (BOPTEST-KPIs auf der Testperiode):'))
        self.kpi_table = QTableWidget()
        self.kpi_table.setMaximumHeight(120)
        layout.addWidget(self.kpi_table)

    def reload_models(self):
        """Neu einlesen, z. B. nachdem ein Training gerade ein Modell gesichert hat."""
        models = sorted(p.stem for p in self.models_dir.glob('*.zip')) if self.models_dir.exists() else []
        previously = {i.text() for i in self.strategy_list.selectedItems()}
        self.strategy_list.blockSignals(True)
        self.strategy_list.clear()
        for name in ['Regel (RBC)'] + models:
            self.strategy_list.addItem(QListWidgetItem(name))
        defaults = previously or ({'Regel (RBC)'} | set(models[-1:]))
        for i in range(self.strategy_list.count()):
            item = self.strategy_list.item(i)
            item.setSelected(item.text() in defaults)
        self.strategy_list.blockSignals(False)
        self._on_selection_changed()

    def _load(self, name: str) -> tuple[pd.DataFrame, dict]:
        mtime = 0.0
        if name != 'Regel (RBC)':
            path = self.models_dir / f'{name}.zip'
            mtime = path.stat().st_mtime if path.exists() else 0.0
        key = (name, mtime)
        if key in self._cache:
            return self._cache[key]
        if name == 'Regel (RBC)':
            result = record('rbc')
        else:
            result = record(ALGOS[name.split('_')[0]].load(str(self.models_dir / name)))
        self._cache[key] = result
        return result

    def _color(self, name: str, chosen: list[str]) -> str:
        if name in COLORS:
            return COLORS[name]
        agents = [n for n in chosen if n not in COLORS]
        return AGENT_PALETTE[agents.index(name) % len(AGENT_PALETTE)] if name in agents else AGENT_COLOR

    def _on_selection_changed(self):
        self.timer.stop()
        self.play_btn.setChecked(False)
        chosen = [i.text() for i in self.strategy_list.selectedItems()]
        if not chosen:
            self._runs, self._kpis = {}, {}
            self.figure.clear()
            self.canvas.draw_idle()
            self.kpi_table.setRowCount(0)
            self.status_label.setText('Mindestens eine Strategie auswählen.')
            return

        self.status_label.setText('Simuliere Testperiode …')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._runs, self._kpis = {}, {}
        errors = []
        try:
            for n in chosen:
                try:
                    df, kpis = self._load(n)
                    self._runs[n] = df
                    self._kpis[n] = kpis
                except requests.exceptions.RequestException:
                    QApplication.restoreOverrideCursor()
                    self.figure.clear()
                    self.canvas.draw_idle()
                    self.kpi_table.setRowCount(0)
                    self.status_label.setText('⚠️ BOPTEST ist unter http://127.0.0.1:8000 nicht erreichbar. '
                                              'Erst scripts\\start_boptest.ps1 ausführen (siehe README.md), '
                                              'dann hier neu auswählen.')
                    return
                except Exception as e:
                    # Häufigste Ursache: ein altes, mit einem inzwischen geänderten Environment
                    # (anderer Beobachtungs-/Aktionsraum, z. B. nach Hinzufügen von Batterie/PV)
                    # trainiertes Modell passt nicht mehr — nicht die ganze Ansicht abbrechen,
                    # sondern nur diese eine Strategie überspringen und klar benennen, welche.
                    errors.append(f'{n}: {e}')
        finally:
            QApplication.restoreOverrideCursor()

        if errors:
            self.status_label.setText('⚠️ Nicht ladbar (vermutlich mit einem älteren Environment-Stand '
                                      'trainiert, siehe workbench.md) — bitte neu trainieren oder aus '
                                      'models/ entfernen: ' + '; '.join(errors))
        if not self._runs:
            self.figure.clear()
            self.canvas.draw_idle()
            self.kpi_table.setRowCount(0)
            return

        n_hours = len(next(iter(self._runs.values())))
        self.hour_slider.blockSignals(True)
        self.hour_slider.setRange(1, max(1, n_hours))
        self.hour_slider.setValue(n_hours)
        self.hour_slider.blockSignals(False)
        self._redraw()
        self._update_kpis()

    def _on_play_toggled(self, checked):
        if checked:
            self.play_btn.setText('⏸ Pause')
            self.timer.start(self.speed_slider.value())
        else:
            self.play_btn.setText('▶ Abspielen')
            self.timer.stop()

    def _advance(self):
        h, n = self.hour_slider.value(), self.hour_slider.maximum()
        self.hour_slider.setValue(1 if h >= n else h + 1)

    def _redraw(self):
        if not self._runs:
            return
        self.timer.setInterval(self.speed_slider.value())
        upto = self.hour_slider.value()
        chosen = list(self._runs)

        self.figure.clear()
        # Fünf Reihen statt einer kombinierten — SOC (0..1, dimensionslos) und Solarleistung
        # (kW) auf eine gemeinsame Achse zu zwingen wäre irreführend (unterschiedliche
        # Einheiten/Skalen), deshalb bekommt jede Größe ihre eigene Achse.
        axes = self.figure.subplots(5, 1, sharex=True, height_ratios=[2.2, 1, 1, 1, 1])
        ax_temp, ax_hp, ax_soc, ax_solar, ax_price = axes
        first = next(iter(self._runs.values())).iloc[:upto]

        ax_temp.plot(first.t, first.setpoint_heat, color='#8a98a3', linestyle='--', lw=1, label='Sollwert Heizen')
        ax_temp.plot(first.t, first.setpoint_cool, color='#8a98a3', linestyle=':', lw=1, label='Sollwert Kühlen')
        ax_temp.fill_between(first.t, first.setpoint_heat, first.setpoint_cool, color='#8a98a3', alpha=.12,
                             label='Komfortband')
        for n in chosen:
            d = self._runs[n].iloc[:upto]
            c = self._color(n, chosen)
            lw = 2.2 if len(chosen) == 1 or n == chosen[-1] else 1.3
            ax_temp.plot(d.t, d.indoor, color=c, lw=lw, label=n)
            ax_hp.plot(d.t, d.heat_pump_action, color=c, lw=lw)
            ax_soc.plot(d.t, d.battery_soc, color=c, lw=lw)

        # Solarerzeugung und Preis hängen nur vom Wetter/Szenario ab, nicht von der gewählten
        # Strategie — deshalb je eine einzelne Linie statt je Strategie überlagert (eine
        # Datenreihe braucht laut Diagramm-Richtlinie keine eigene Legende, der Achsentitel reicht).
        ax_solar.plot(first.t, first.solar_power, color=SOLAR_COLOR, lw=1.6)
        ax_price.plot(first.t, first.price, color=PRICE_COLOR, lw=1.6)

        ax_temp.set_ylabel('Zone (°C)', fontsize=8)
        ax_hp.set_ylabel('Wärmepumpe (0–1)', fontsize=8)
        ax_soc.set_ylabel('Batterie-SOC (0–1)', fontsize=8)
        ax_solar.set_ylabel('Solar (kW)', fontsize=8)
        ax_price.set_ylabel('Preis ($/kWh)', fontsize=8)
        ax_price.set_xlabel('Stunde der Testperiode')
        for ax in axes:
            ax.tick_params(labelsize=7)

        # Eine gemeinsame Legende oberhalb aller Reihen statt mehrerer Einzel-Legenden, die
        # sonst Datenlinien verdecken — deckt Referenzlinien (Sollwerte/Komfortband) und alle
        # Strategiefarben ab, gilt sinngemäß auch für die Wärmepumpen-/SOC-Reihen darunter,
        # die dieselbe Farbzuordnung je Strategie verwenden.
        handles, labels = ax_temp.get_legend_handles_labels()
        self.figure.legend(handles, labels, loc='outside upper center', fontsize=7,
                           ncol=min(len(handles), 5), frameon=True)
        self.canvas.draw_idle()

        lines = []
        for n in chosen:
            r = self._runs[n].iloc[upto - 1]
            inside = r.setpoint_heat <= r.indoor <= r.setpoint_cool
            lines.append(f"{n}: {r.indoor:.1f}°C · Wärmepumpe {r.heat_pump_action:.2f} · "
                        f"Batterie {r.battery_power:+.1f} kW (SOC {r.battery_soc:.2f}) · "
                        + ('im Band' if inside else 'außerhalb Band'))
        r0 = first.iloc[-1]
        self.status_label.setText(f"Stunde {int(r0.hour):02d}:00 — Solar {r0.solar_power:.2f} kW — "
                                  + '   |   '.join(lines))

    def _update_kpis(self):
        names = list(self._kpis)
        keys = [k for k in KPI_LABELS if all(k in self._kpis[n] for n in names)]
        self.kpi_table.setRowCount(len(names))
        self.kpi_table.setColumnCount(len(keys))
        self.kpi_table.setHorizontalHeaderLabels([KPI_LABELS[k] for k in keys])
        self.kpi_table.setVerticalHeaderLabels(names)
        for i, n in enumerate(names):
            for j, k in enumerate(keys):
                v = self._kpis[n][k]
                self.kpi_table.setItem(i, j, QTableWidgetItem(f'{v:.4f}' if v is not None else '—'))
        self.kpi_table.resizeColumnsToContents()
