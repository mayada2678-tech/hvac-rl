"""Einheitliches Erscheinungsbild — an VS Code Dark+ angelehnt (dunkles Grau, Akzentblau,
Segoe UI). Ein QSS-Stylesheet fürs ganze Fenster plus passende matplotlib-rcParams, damit
Diagramme sich nahtlos einfügen statt wie ein weißer Fremdkörper im dunklen Fenster zu wirken.

apply(app) einmal direkt nach dem Erstellen der QApplication aufrufen (siehe app.py).
"""
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# VS Code Dark+ Palette (siehe code.visualstudio.com/api/references/theme-color)
BG = '#1e1e1e'            # Editor-Hintergrund
PANEL = '#252526'          # Seitenleisten/Panels
PANEL_ALT = '#2d2d2d'       # leicht abgesetzte Flächen (inaktive Tabs, Kopfzeilen)
INPUT_BG = '#3c3c3c'         # Eingabefelder
BORDER = '#3c3c3c'
BORDER_LIGHT = '#454545'
TEXT = '#cccccc'
TEXT_MUTED = '#9d9d9d'
ACCENT = '#0e639c'           # Button-Blau
ACCENT_HOVER = '#1177bb'
ACCENT_FOCUS = '#007fd4'
SELECTION = '#264f78'
GOOD = '#89d185'
WARN = '#cca700'
BAD = '#f14c4c'

FONT_FAMILY = 'Segoe UI'
FONT_SIZE = 10

# Kategoriale Diagrammfarben (feste Reihenfolge je Entität, nicht nach Rang/Auswahl) —
# hell/gesättigt genug, um auf dem dunklen BG (#1e1e1e) gut lesbar zu sein.
PLOT_PALETTE = ['#4FC1FF', '#4EC9B0', '#DCDCAA', '#F48771', '#9CDCFE', '#C586C0', '#B5CEA8', '#D7BA7D']

QSS = f"""
* {{
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE}pt;
    color: {TEXT};
}}
QWidget {{
    background-color: {BG};
}}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {BG};
    border: none;
}}
QLabel {{
    background: transparent;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {BG};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {PANEL_ALT};
    color: {TEXT_MUTED};
    padding: 7px 16px;
    border: 1px solid {BORDER};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background-color: {BG};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT_FOCUS};
}}
QTabBar::tab:hover:!selected {{
    background-color: #333333;
    color: {TEXT};
}}

QPushButton {{
    background-color: #3a3d41;
    color: {TEXT};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 3px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: #45494e;
}}
QPushButton:pressed {{
    background-color: #333333;
}}
QPushButton:disabled {{
    color: #6a6a6a;
    background-color: #2d2d2d;
}}
QPushButton:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT_FOCUS};
}}
QPushButton[accent="true"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT_FOCUS};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton[danger="true"] {{
    background-color: #5a1d1d;
    border: 1px solid #832626;
}}
QPushButton[danger="true"]:hover {{
    background-color: #7a2727;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {INPUT_BG};
    color: {TEXT};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: {SELECTION};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT_FOCUS};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {INPUT_BG};
    color: {TEXT};
    selection-background-color: {SELECTION};
    border: 1px solid {BORDER_LIGHT};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: #454545;
    width: 14px;
    border: none;
}}

QGroupBox {{
    border: 1px solid {BORDER_LIGHT};
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 6px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {TEXT};
}}
QRadioButton, QCheckBox {{
    spacing: 6px;
    background: transparent;
}}
QRadioButton::indicator, QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_LIGHT};
    background-color: {INPUT_BG};
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator {{
    border-radius: 3px;
}}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {{
    background-color: {ACCENT_FOCUS};
    border-color: {ACCENT_FOCUS};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {INPUT_BG};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_FOCUS};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

QListWidget, QTableWidget {{
    background-color: {PANEL};
    alternate-background-color: #2a2a2a;
    color: {TEXT};
    border: 1px solid {BORDER};
    gridline-color: {BORDER};
    selection-background-color: {SELECTION};
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background-color: {PANEL_ALT};
    color: {TEXT_MUTED};
    padding: 4px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
}}

QScrollBar:vertical {{
    background: {BG};
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background: #4a4a4a;
    min-height: 24px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: #5a5a5a;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {BG};
    height: 12px;
}}
QScrollBar::handle:horizontal {{
    background: #4a4a4a;
    min-width: 24px;
    border-radius: 5px;
}}

QFrame[card="true"] {{
    background-color: {PANEL};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 6px;
}}

QToolTip {{
    background-color: #2d2d2d;
    color: {TEXT};
    border: 1px solid {BORDER_LIGHT};
    padding: 4px;
}}
"""


def apply(app: QApplication):
    app.setStyle('Fusion')
    app.setFont(QFont(FONT_FAMILY, FONT_SIZE))
    app.setStyleSheet(QSS)
    _apply_matplotlib_theme()


def _apply_matplotlib_theme():
    import matplotlib
    matplotlib.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': BG,
        'savefig.facecolor': BG,
        'axes.edgecolor': BORDER_LIGHT,
        'axes.labelcolor': TEXT,
        'axes.titlecolor': TEXT,
        'axes.grid': True,
        'grid.color': BORDER,
        'grid.alpha': 0.6,
        'grid.linewidth': 0.6,
        'text.color': TEXT,
        'xtick.color': TEXT_MUTED,
        'ytick.color': TEXT_MUTED,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.facecolor': PANEL,
        'legend.edgecolor': BORDER_LIGHT,
        'legend.labelcolor': TEXT,
        'legend.fontsize': 8,
        'legend.framealpha': 0.92,
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT_FAMILY, 'Segoe UI', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.prop_cycle': matplotlib.cycler(color=PLOT_PALETTE),
        'figure.dpi': 100,
    })
