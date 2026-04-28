"""
Neon-style Qt stylesheet (QSS) strings and palette for the UrbanQuality-AI GUI.
"""

NEON_STYLE = """
QMainWindow {
    background-color: #050505;
}

QFrame#Sidebar {
    background-color: #0a0a0a;
    border-right: 1px solid #00f2ff;
}

QFrame#PanelBackground {
    background-color: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-radius: 10px;
}

QSplitter::handle {
    background-color: #1a1a1a;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:hover {
    background-color: #00f2ff;
}

QLabel {
    color: #00f2ff;
    font-family: 'Segoe UI', sans-serif;
    text-transform: uppercase;
    font-weight: bold;
    font-size: 10px;
}

QLabel#PanelTitle {
    font-size: 14px;
    letter-spacing: 2px;
    margin-bottom: 8px;
    color: #fff;
    border-bottom: 1px solid #00f2ff;
    padding-bottom: 5px;
}

QLabel#PanelSubtitle {
    font-size: 11px;
    font-weight: normal;
    text-transform: none;
    letter-spacing: 0px;
    color: #9cf;
    margin-top: 0px;
    margin-bottom: 12px;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background-color: #111;
    border: 1px solid #333;
    color: #fff;
    padding: 8px;
    border-radius: 2px;
}

QAbstractSpinBox {
    /* Reserve space for the right-side buttons so clicks don't land on the editor area. */
    padding-right: 24px;
}

QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    subcontrol-origin: padding;
    width: 18px;
    border-left: 1px solid #333;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 10px;
    height: 10px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 7px solid #0b7a2a;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 10px;
    height: 10px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid #8a1f1f;
}

QAbstractSpinBox::up-button {
    subcontrol-position: top right;
    height: 14px;
    background-color: #151515;
    border-bottom: 1px solid #262626;
}

QAbstractSpinBox::down-button {
    subcontrol-position: bottom right;
    height: 14px;
    background-color: #151515;
    border-top: 1px solid #262626;
}

QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {
    border-left: 1px solid #00f2ff;
    background-color: #1f1f1f;
}

QLineEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #00f2ff;
}

QPushButton#SidebarBtn {
    background-color: transparent;
    border: none;
    color: #444;
    padding: 15px;
    font-size: 20px;
}

QPushButton#SidebarBtn:hover {
    color: #00f2ff;
}

QPushButton#SidebarBtn:checked {
    color: #00f2ff;
    background-color: #001a1a;
    border-left: 3px solid #00f2ff;
}

QPushButton#RunBtn {
    background-color: #00f2ff;
    color: #000;
    font-weight: bold;
    font-size: 11px;
    padding: 12px;
    border-radius: 5px;
    margin-top: 10px;
}

QPushButton#RunBtn:hover {
    background-color: #55faff;
}

QPushButton#RunBtn:disabled {
    background-color: #333;
    color: #666;
}

QTextEdit#Console {
    background-color: #050505;
    border: 1px solid #1a1a1a;
    color: #00ff41;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}

QProgressBar {
    border: 1px solid #1a1a1a;
    background-color: #0a0a0a;
    text-align: center;
    color: #fff;
    height: 10px;
}

QProgressBar::chunk {
    background-color: #00f2ff;
}

QFrame#MapLayerPanel {
    background-color: rgba(10, 10, 10, 220);
    border: 1px solid rgba(0, 242, 255, 0.28);
    border-radius: 8px;
    min-width: 168px;
}

QLabel#MapLayerTitle {
    color: #fff;
    font-size: 11px;
    letter-spacing: 1px;
    padding: 6px 8px 2px 8px;
}

QListWidget#MapLayerList {
    background-color: #050505;
    border: none;
    color: #ccc;
    font-size: 11px;
    outline: none;
}

QListWidget#MapLayerList::item {
    padding: 6px 4px;
    border-radius: 4px;
}

QListWidget#MapLayerList::item:selected {
    background-color: #001a1a;
    color: #00f2ff;
}

QListWidget#MapLayerList::item:hover:!active {
    background-color: #111;
}

QPushButton#LayerOrderUp, QPushButton#LayerOrderDown {
    background-color: #111;
    border: 1px solid #333;
    color: #00f2ff;
    font-size: 10px;
    padding: 4px 6px;
    border-radius: 3px;
}

QPushButton#LayerOrderUp:hover, QPushButton#LayerOrderDown:hover {
    border: 1px solid #00f2ff;
}
"""
