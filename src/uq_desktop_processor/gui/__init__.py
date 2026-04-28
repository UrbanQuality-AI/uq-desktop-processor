"""
GUI module:   init  .
"""

import ctypes
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .paths import APP_ICON_PATH
from .platform.webengine import configure_webengine_for_deck_gl
from .shell.explorer import UrbanQualityAIExplorer


def main() -> int:
    """
    Run main.

    :return: Result of this step or updated UI/application state.

    Example::
        In: main()
        Out: UI/application state updated as intended.
    """
    configure_webengine_for_deck_gl()
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("uq_desktop_processor.app")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app_icon = QIcon(str(APP_ICON_PATH))
    app.setWindowIcon(app_icon)
    gui = UrbanQualityAIExplorer()
    gui.setWindowIcon(app_icon)
    gui.show()
    return app.exec()


__all__ = ["main", "UrbanQualityAIExplorer"]
