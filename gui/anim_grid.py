"""Mehrere Animationen (web/hvac-agent-animation.html, eingebettet) nebeneinander — eine je
Strategie bzw. Trainingslauf, damit sich Methoden direkt vergleichen lassen. Genutzt von
gui/watch_view.py ("Beobachten") und gui/agent_view.py ("Training").
"""
import json
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

ANIM_PAGE = Path(__file__).resolve().parent.parent / 'web' / 'hvac-agent-animation.html'
# Jede eingebettete Webansicht ist ein eigener Browser-Prozess (~100 MB) — mehr als vier
# nebeneinander wären ohnehin zu schmal zum Lesen.
MAX_ANIMATIONS = 4


class _AnimPanel(QWidget):
    """Eine Animation mit farbiger Überschrift. Zustände, die vor dem Laden der Seite
    ankommen, werden gemerkt und nach dem Laden nachgereicht."""

    def __init__(self, name: str, color: str | None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.title = QLabel()
        layout.addWidget(self.title)
        self.set_title(name, color)
        self.view = QWebEngineView()
        url = QUrl.fromLocalFile(str(ANIM_PAGE))
        url.setQuery('embedded=1')
        self._loaded = False
        self._pending = None
        self.view.loadFinished.connect(self._on_loaded)
        self.view.setUrl(url)
        layout.addWidget(self.view, 1)

    def set_title(self, name: str, color: str | None):
        dot = f'<span style="color:{color}">●</span> ' if color else ''
        self.title.setText(f'{dot}<b>{name}</b>')

    def _on_loaded(self, _ok):
        self._loaded = True
        if self._pending is not None:
            self.push(self._pending)

    def push(self, state: dict):
        if not self._loaded:
            self._pending = state
            return
        self.view.page().runJavaScript(f'window.applyState && window.applyState({json.dumps(state)})')


class AnimGrid(QWidget):
    """Hält je Name eine Animation; set_names() legt fehlende an und entfernt überzählige,
    push() gibt einer davon einen neuen Zustand."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._panels: dict[str, _AnimPanel] = {}
        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet('color: #9d9d9d;')
        self._note.hide()

    def set_names(self, names: list[str], colors: dict[str, str] | None = None):
        colors = colors or {}
        shown = names[:MAX_ANIMATIONS]
        for name in list(self._panels):
            if name not in shown:
                panel = self._panels.pop(name)
                self._layout.removeWidget(panel)
                panel.deleteLater()
        for name in shown:
            if name not in self._panels:
                self._panels[name] = _AnimPanel(name, colors.get(name))
            else:
                self._panels[name].set_title(name, colors.get(name))
        # Reihenfolge wie übergeben (Auswahl-/Startreihenfolge), damit Farben/Plätze stabil bleiben
        for panel in self._panels.values():
            self._layout.removeWidget(panel)
        for name in shown:
            self._layout.addWidget(self._panels[name], 1)
        self._layout.removeWidget(self._note)
        if len(names) > MAX_ANIMATIONS:
            self._note.setText(f'Animation für die ersten {MAX_ANIMATIONS} von {len(names)}.')
            self._layout.addWidget(self._note)
            self._note.show()
        else:
            self._note.hide()

    def push(self, name: str, state: dict):
        panel = self._panels.get(name)
        if panel is not None:
            panel.push(state)
