"""Hauptfenster. Kein Browser, kein Server — ein gewöhnliches Qt-Programmfenster (PySide6)."""
from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QTabWidget

from gui.agent_view import AgentView
from gui.dataset_view import DatasetView


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.setWindowTitle('HVAC-RL — BOPTEST')
        self.resize(1400, 900)

        tabs = QTabWidget()
        tabs.addTab(AgentView(root), '🤖 Agent')
        tabs.addTab(DatasetView(), '📊 Datensatz')
        self.setCentralWidget(tabs)
