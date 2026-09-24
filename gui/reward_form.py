"""Qt-Formular "Belohnungsfunktion": jeder Parameter von logic/reward.py::RewardWrapper
einzeln einstellbar, vorbelegt mit der besten Schätzung aus Literatur/Datenblättern
(logic/reward.py::REWARD_DEFAULTS, Quelle je Feld im Tooltip). Zeigt die Formel und rechnet
live aus, was die eingestellten Werte konkret bedeuten (z. B. "1 K·h ≙ 192 €").
"""
import re

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout)

from logic.battery_env import MAX_SOC, MIN_SOC
from logic.reward import COMFORT_PRESETS, PARAM_INFO, REWARD_DEFAULTS, wear_cost_per_kwh

# Standard der Komfortgewicht-Studie: BOPTEST-Gym-Standard (1) bis kostenbetont (0.1).
DEFAULT_STUDY_WEIGHTS = '1.0, 0.3, 0.1'

FORMULA = ('r = −scale · [ (Kosten + Verschleiß − Restwert) / Wohnfläche '
           '+ w · ΔKomfort-Defizit ]')


class RewardForm(QGroupBox):
    """Alle Belohnungsparameter — gilt für alle Algorithmen, damit sie vergleichbar bleiben."""

    def __init__(self, parent=None):
        super().__init__('🎯 Belohnungsfunktion (für alle Algorithmen)', parent)
        self._fields: dict[str, QDoubleSpinBox] = {}
        outer = QVBoxLayout(self)

        formula = QLabel(f'<b>{FORMULA}</b><br><small>Kosten, Verschleiß, Restwert in €; '
                         'Komfort-Defizit in Kelvin-Stunden (BOPTEST-KPI tdis_tot); Restwert nur im '
                         'letzten Schritt. Details: logic/reward.py. Maus über ein Feld = Quelle '
                         'des Standardwerts.</small>')
        formula.setWordWrap(True)
        outer.addWidget(formula)

        self.preset = QComboBox()
        self.preset.addItems(list(COMFORT_PRESETS))
        self.preset.addItem('Eigene Einstellung')
        self.preset.setToolTip('Komfortgewicht-Voreinstellungen aus BOPTEST-Gym (Arroyo et al.).')
        self.preset.currentTextChanged.connect(self._on_preset)
        outer.addWidget(self.preset)

        form = QFormLayout()
        outer.addLayout(form)
        for key, meta in PARAM_INFO.items():
            spin = QDoubleSpinBox()
            spin.setRange(meta['lo'], meta['hi'])
            spin.setSingleStep(meta['step'])
            spin.setDecimals(meta['decimals'])
            spin.setToolTip(meta['help'])
            spin.valueChanged.connect(self._update_summary)
            label = QLabel(meta['label'])
            label.setToolTip(meta['help'])
            label.setWordWrap(True)
            form.addRow(label, spin)
            self._fields[key] = spin
        self._fields['w_comfort'].valueChanged.connect(self._sync_preset)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)

        # Komfortgewicht-Studie: dieselbe Belohnung, nur w variiert -> eine Kosten-Komfort-Kurve
        # statt eines Einzelergebnisses (Auswertung im Reiter "Vergleich").
        self.study_check = QCheckBox('Komfortgewicht-Studie: je w eine Variante trainieren')
        self.study_check.setToolTip('Trainiert je gewähltem Algorithmus eine Variante pro w-Wert '
                                    '(Modelle z. B. sac_seed0_w0_3). Im Reiter „Vergleich“ entsteht daraus '
                                    'die Kosten-Komfort-Kurve gegenüber dem RBC.')
        self.study_weights = QLineEdit(DEFAULT_STUDY_WEIGHTS)
        self.study_weights.setPlaceholderText('z. B. 1.0, 0.3, 0.1')
        self.study_weights.setEnabled(False)
        self.study_check.toggled.connect(self._on_study_toggled)
        outer.addWidget(self.study_check)
        form_study = QFormLayout()
        form_study.addRow('w-Werte:', self.study_weights)
        outer.addLayout(form_study)

        reset = QPushButton('↺ Beste theoretische Werte')
        reset.setToolTip('Alle Felder auf die Standardwerte aus logic/reward.py zurücksetzen.')
        reset.clicked.connect(self.reset_defaults)
        outer.addWidget(reset)

        self.reset_defaults()

    def reset_defaults(self):
        for key, value in REWARD_DEFAULTS.items():
            self._fields[key].setValue(float(value))
        self._sync_preset()
        self._update_summary()

    def _on_study_toggled(self, on: bool):
        self.study_weights.setEnabled(on)
        # In der Studie kommt w aus der Liste, nicht aus dem Einzelfeld.
        self._fields['w_comfort'].setEnabled(not on)
        self.preset.setEnabled(not on)

    def study_weight_list(self) -> list[float] | None:
        """w-Werte der Studie, oder None, wenn keine Studie gewählt ist. ValueError mit
        verständlicher Meldung bei ungültiger Eingabe."""
        if not self.study_check.isChecked():
            return None
        # Dezimalkomma zulassen ("1,0, 0,3, 0,1"): Komma zwischen zwei Ziffern = Dezimalzeichen,
        # alle übrigen Kommas/Semikolons/Leerzeichen trennen die Werte.
        text = re.sub(r'(\d),(\d)', r'\1.\2', self.study_weights.text().strip())
        try:
            weights = [float(p) for p in re.split(r'[\s,;]+', text) if p]
        except ValueError:
            raise ValueError('Die w-Werte bitte als Zahlen eingeben, getrennt durch Komma oder '
                             'Semikolon, z. B. 1.0, 0.3, 0.1 oder 1,0; 0,3; 0,1')
        weights = list(dict.fromkeys(weights))   # Doppelte entfernen, Reihenfolge behalten
        if not weights or any(w < 0 for w in weights):
            raise ValueError('Mindestens ein w-Wert, keiner negativ.')
        return weights

    def _on_preset(self, text: str):
        if text in COMFORT_PRESETS:
            self._fields['w_comfort'].setValue(COMFORT_PRESETS[text])

    def _sync_preset(self):
        w = self._fields['w_comfort'].value()
        match = next((name for name, v in COMFORT_PRESETS.items() if abs(v - w) < 1e-9), 'Eigene Einstellung')
        self.preset.blockSignals(True)
        self.preset.setCurrentText(match)
        self.preset.blockSignals(False)

    def _update_summary(self):
        v = self.values()
        wear = wear_cost_per_kwh(v['battery_invest_eur_per_kwh'], v['battery_cycle_life'], MAX_SOC - MIN_SOC)
        eur_per_kh = v['w_comfort'] * v['floor_area_m2']
        self.summary.setText(
            f'<small><b>Bedeutet konkret:</b> 1 Kelvin-Stunde Komfortverletzung ≙ '
            f'{eur_per_kh:,.{0 if eur_per_kh >= 10 else 2}f} € Stromkosten · Batterie-Verschleiß {wear * 100:.2f} ct je kWh '
            f'Durchsatz (Entladetiefe {MAX_SOC - MIN_SOC:.0%}) · Restwert zu {v["w_terminal"]:.0%} '
            f'angerechnet.</small>'.replace(',', ' '))

    def values(self) -> dict:
        v = {key: spin.value() for key, spin in self._fields.items()}
        v['battery_cycle_life'] = int(v['battery_cycle_life'])
        return v
