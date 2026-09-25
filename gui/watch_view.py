"""Qt-Baustein "Beobachten": Stunde-für-Stunde-Animation der Testperiode (3 Tage im Februar,
siehe logic/envs.py) — Zonentemperatur vs. Komfortband, Wärmepumpen-Modulation, dynamischer
Strompreis, Belohnung je Schritt aufgeteilt in ihre vier Anteile. Strategien: BOPTESTs
eingebauter Regler (RBC) und beliebig viele trainierte Modelle, zum direkten Vergleich.

Zwei Unterreiter, beide vom selben Schieberegler/Abspielen gesteuert: "Diagramme"
(matplotlib) und "Animation" (web/hvac-agent-animation.html in einer eingebetteten
Webansicht, bekommt je Stunde den Zustand der zuletzt gewählten Strategie).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from stable_baselines3 import PPO, SAC, TD3

from logic.envs import STEP_PERIOD, TEST_START
from logic.evaluation import obs_normalized
from logic.watch import record

ANIM_PAGE = Path(__file__).resolve().parent.parent / 'web' / 'hvac-agent-animation.html'

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
# Anteile der Belohnung (logic/reward.py::REWARD_PARTS) — eigene, feste Farben,
# getrennt von den Strategiefarben, weil das Diagramm nur eine Strategie zeigt.
REWARD_PART_STYLE = [
    ('reward_cost', 'Stromkosten', '#569CD6'),
    ('reward_comfort', 'Komfort', '#D16969'),
    ('reward_battery', 'Verschleiß', '#6A9955'),
    ('reward_terminal', 'Restwert', '#E2C08D'),
]

KPI_LABELS = {
    'cost_tot': 'Kosten (€ bzw. $/m²)', 'tdis_tot': 'Komfort-Defizit (Kh/m²)',
    'idis_tot': 'Unbehaglichkeits-Index', 'ener_tot': 'Energie (kWh/m²)',
    'emis_tot': 'CO2-Emissionen (kg/m²)', 'time_rat': 'Rechenzeit-Verhältnis',
}


def plot_reward_parts(ax, x, d: pd.DataFrame, width: float, title: str | None = None,
                      clip_terminal: bool = True, legend: bool = True):
    """Belohnung als gestapelte Balken aus ihren Anteilen (logic/reward.py): negative Anteile
    nach unten, positive nach oben gestapelt, Summe = r. Auch vom Training-Reiter genutzt
    (gui/agent_view.py, dort ein Balken je Auswertung)."""
    parts = [(k, label, c) for k, label, c in REWARD_PART_STYLE if k in d]
    if not parts:
        return
    pos = np.zeros(len(d))
    neg = np.zeros(len(d))
    for key, label, color in parts:
        v = d[key].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), width=width, color=color,
               edgecolor=ax.get_facecolor(), linewidth=0.4, label=label)
        pos += np.clip(v, 0, None)
        neg += np.clip(v, None, 0)
    ax.axhline(0, color='#8a98a3', lw=0.6)

    # Je Stunde: der Restwert fällt nur im letzten Schritt an und ist meist viel größer als ein
    # einzelner Stundenanteil — die Skala nach den laufenden Anteilen richten, damit die
    # stündlichen Balken lesbar bleiben, und einen abgeschnittenen Restwert beschriften.
    running = d[[k for k, *_ in parts if k != 'reward_terminal']].fillna(0.0).to_numpy(dtype=float)
    if clip_terminal and running.size:
        lo = min(np.clip(running, None, 0).sum(axis=1).min(), 0.0)
        hi = max(np.clip(running, 0, None).sum(axis=1).max(), 0.0)
        pad = 0.15 * (hi - lo) or 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        if 'reward_terminal' in d:
            term = d['reward_terminal'].fillna(0.0).to_numpy(dtype=float)
            i = int(np.flatnonzero(term)[-1]) if term.any() else None
            if i is not None and not (lo - pad <= neg[i] and pos[i] <= hi + pad):
                ax.annotate(f'Restwert {term[i]:+.4f}', xy=(x[i], hi if term[i] > 0 else lo),
                            xytext=(-4, 0), textcoords='offset points', ha='right', va='center',
                            fontsize=7, color='#cccccc')
    if legend:
        # Neben statt im Diagramm: die Balken füllen die ganze Höhe, eine Legende darin verdeckt sie.
        ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1.0), fontsize=6.5, frameon=True,
                  title=title, title_fontsize=6.5)


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

        self.figure = Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        # Mindesthöhe erzwingen: bei zu wenig Platz (kleines Fenster) würde Qt die Zeichenfläche
        # sonst über die Lesbarkeit hinaus stauchen und Achsenbeschriftungen/Legende würden
        # sich wieder überlappen — lieber im Zweifel scrollen als unleserlich werden.
        self.canvas.setMinimumHeight(580)
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self.canvas, 'Diagramme')
        self.anim_view = QWebEngineView()
        url = QUrl.fromLocalFile(str(ANIM_PAGE))
        url.setQuery('embedded=1')
        self.anim_view.loadFinished.connect(lambda _ok: self._push_anim_state())
        self.anim_view.setUrl(url)
        self.view_tabs.addTab(self.anim_view, 'Animation')
        layout.addWidget(self.view_tabs, 1)

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
            path = self.models_dir / f'{name}.zip'
            result = record(ALGOS[name.split('_')[0]].load(str(self.models_dir / name)),
                            normalize=obs_normalized(path))
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
        # Vier Reihen, jede mit genau einer Einheit/Skala (nie zwei Skalen auf einer Achse):
        # Wärmepumpe und Batterie-SOC sind beide dimensionslose 0..1-Größen und teilen sich
        # deshalb legitim eine Achse, unterschieden per Linienstil statt Farbe (Farbe bleibt
        # für die Strategie reserviert). Solar (kW) und Preis ($/kWh) bekommen je eine eigene
        # Achse — sie zusammen auf eine Achse zu zwingen (Zwei-Skalen-Diagramm) wäre irreführend.
        # Fünfte Reihe: Belohnungsanteile als gestapelte Balken, nur für die hervorgehobene
        # (zuletzt gewählte) Strategie — Stapel mehrerer Strategien übereinander wären unlesbar.
        axes = self.figure.subplots(5, 1, sharex=True, height_ratios=[2.2, 1, 1, 1, 1.2])
        ax_temp, ax_hp, ax_solar, ax_price, ax_rew = axes
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
            ax_hp.plot(d.t, d.heat_pump_action, color=c, lw=lw, linestyle='-')
            ax_hp.plot(d.t, d.battery_soc, color=c, lw=max(1.0, lw - 0.6), linestyle='--')
        # Referenz-Linienstile fürs Wärmepumpen-/SOC-Diagramm einmal neutral in der Legende
        # erklären (Farbe = Strategie, Linienstil = Größe — zusammen macht das die Reihe
        # eindeutig, ohne je Strategie zwei weitere Legendeneinträge zu brauchen).
        ax_hp.plot([], [], color='#8a98a3', linestyle='-', lw=1.4, label='Wärmepumpe')
        ax_hp.plot([], [], color='#8a98a3', linestyle='--', lw=1.4, label='Batterie-SOC')

        # Solarerzeugung und Preis hängen nur vom Wetter/Szenario ab, nicht von der gewählten
        # Strategie — deshalb je eine einzelne Linie statt je Strategie überlagert (eine
        # Datenreihe braucht laut Diagramm-Richtlinie keine eigene Legende, der Achsentitel reicht).
        ax_solar.plot(first.t, first.solar_power, color=SOLAR_COLOR, lw=1.6)
        ax_price.plot(first.t, first.price, color=PRICE_COLOR, lw=1.6)
        self._draw_reward_parts(ax_rew, chosen[-1], upto)

        ax_temp.set_ylabel('Zone (°C)', fontsize=8)
        ax_hp.set_ylabel('Wärmepumpe / SOC (0–1)', fontsize=8)
        ax_solar.set_ylabel('Solar (kW)', fontsize=8)
        ax_price.set_ylabel('Preis ($/kWh)', fontsize=8)
        ax_rew.set_ylabel('Belohnung', fontsize=8)
        ax_rew.set_xlabel('Stunde der Testperiode')
        for ax in axes:
            ax.tick_params(labelsize=7)

        # Eine gemeinsame Legende oberhalb aller Reihen statt mehrerer Einzel-Legenden, die
        # sonst Datenlinien verdecken — deckt Referenzlinien (Sollwerte/Komfortband), alle
        # Strategiefarben und die beiden Linienstile (Wärmepumpe/SOC) ab.
        handles, labels = ax_temp.get_legend_handles_labels()
        h2, l2 = ax_hp.get_legend_handles_labels()
        handles += h2[-2:]; labels += l2[-2:]   # nur die zwei neutralen Stil-Erklärungen dazu
        self.figure.legend(handles, labels, loc='outside upper center', fontsize=7,
                           ncol=min(len(handles), 4), frameon=True)
        self.canvas.draw_idle()

        lines = []
        for n in chosen:
            r = self._runs[n].iloc[upto - 1]
            inside = r.setpoint_heat <= r.indoor <= r.setpoint_cool
            lines.append(f"{n}: {r.indoor:.1f}°C · Wärmepumpe {r.heat_pump_action:.2f} · "
                        f"Batterie {r.battery_power:+.1f} kW (SOC {r.battery_soc:.2f}) · "
                        + ('im Band' if inside else 'außerhalb Band') + f" · r {r.reward:+.4f}")
        r0 = first.iloc[-1]
        self.status_label.setText(f"Stunde {int(r0.hour):02d}:00 — Solar {r0.solar_power:.2f} kW — "
                                  + '   |   '.join(lines))
        self._push_anim_state()

    def _push_anim_state(self):
        """Zustand der aktuellen Stunde an die eingebettete Animation geben — für die zuletzt
        gewählte Strategie, wie die Belohnungsbalken. Feldnamen wie in
        logic/watch.py::record()s anim_state.json, damit dieselbe Seite beides kann."""
        if not self._runs:
            return
        name = list(self._runs)[-1]
        d = self._runs[name]
        i = min(max(self.hour_slider.value(), 1), len(d)) - 1
        r = d.iloc[i]
        g = lambda key: r[key] if key in r else None
        rbc = name == 'Regel (RBC)'
        state = {
            'strategy': name, 'step': i + 1, 'n_steps': len(d), 'done': i + 1 == len(d),
            'time_s': TEST_START + (int(r.t) + 1) * STEP_PERIOD,
            'price': g('price'), 'price_levels': [d.price.quantile(1 / 3), d.price.quantile(2 / 3)],
            'battery_soc': g('battery_soc'), 'battery_power_kw': g('battery_power'),
            'solar_power_kw': g('solar_power'), 'grid_power_kw': g('grid_power'),
            'heat_pump_power_kw': g('heat_pump_power'),
            'reaTZon_y': r.indoor + 273.15, 'indoor_c': r.indoor,
            'setpoint_heat_c': r.setpoint_heat, 'setpoint_cool_c': r.setpoint_cool, 'outdoor_c': r.outdoor,
            'action': [] if rbc else [r.heat_pump_action, r.battery_power],
            'reward': g('reward'), 'updated': i,
        }
        clean = lambda v: None if isinstance(v, float) and not math.isfinite(v) else v
        state = {k: ([clean(float(x)) for x in v] if isinstance(v, list) else
                     clean(float(v)) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v)
                 for k, v in state.items()}
        self.anim_view.page().runJavaScript(f'window.applyState && window.applyState({json.dumps(state)})')

    def _draw_reward_parts(self, ax, name: str, upto: int):
        d = self._runs[name].iloc[:upto]
        plot_reward_parts(ax, d.t.to_numpy(), d, width=0.85, title=name)

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
