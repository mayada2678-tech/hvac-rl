"""Start: python app.py

Voraussetzung: BOPTEST muss laufen (scripts\\start_boptest.ps1), siehe README.md.

Desktop-Oberfläche (PySide6/Qt) — kein Browser, kein lokaler Server, ein gewöhnliches
Programmfenster: SAC/PPO/TD3 trainieren (Start/Pause/Weiter/Stopp, Hyperparameter,
Lernkurve) und den Agenten auf der Testperiode beobachten (Animation, KPI-Vergleich mit
BOPTESTs eingebautem Regler). Siehe gui/agent_view.py, gui/watch_view.py.

Die eigentliche Logik (BOPTEST-Anbindung, Training, Belohnung) liegt in logic/ und hat keine
Qt-Abhängigkeit — siehe workbench.md.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# `python app.py` in einem Terminal ohne aktivierte Umgebung nimmt das System-Python, dem
# PySide6 & Co. fehlen ("No module named 'PySide6'"). Dann einfach mit dem Python der
# Projektumgebung .venv neu starten, statt abzubrechen.
_VENV_PYTHON = Path(__file__).resolve().parent / '.venv' / 'Scripts' / 'python.exe'
if importlib.util.find_spec('PySide6') is None and _VENV_PYTHON.exists() \
        and Path(sys.prefix).resolve() != _VENV_PYTHON.parent.parent.resolve():
    print(f'Hinweis: Terminal ohne aktivierte .venv ({sys.executable}) - '
          f'starte die App mit der Projektumgebung .venv. Das Fenster erscheint gleich.')
    sys.exit(subprocess.call([str(_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]))

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
