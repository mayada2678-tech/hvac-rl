"""Start: python app.py

Voraussetzung: BOPTEST muss laufen (scripts\\start_boptest.ps1), siehe README.md.

Desktop-Oberfläche (PySide6/Qt) — kein Browser, kein lokaler Server, ein gewöhnliches
Programmfenster: SAC/PPO/TD3 trainieren (Start/Pause/Weiter/Stopp, Hyperparameter,
Lernkurve) und den Agenten auf der Testperiode beobachten (Animation, KPI-Vergleich mit
BOPTESTs eingebautem Regler). Siehe gui/agent_view.py, gui/watch_view.py.

Die eigentliche Logik (BOPTEST-Anbindung, Training, Belohnung) liegt in logic/ und hat keine
Qt-Abhängigkeit — siehe workbench.md.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_API', 'pyside6')
import matplotlib  # noqa: E402
matplotlib.use('QtAgg')

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import theme  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('HVAC-RL')
    theme.apply(app)   # VS-Code-Dark+-Optik fürs ganze Fenster + passende Diagrammfarben
    window = MainWindow(ROOT)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
