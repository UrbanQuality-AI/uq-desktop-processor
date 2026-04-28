"""
QThread-based workers for running pipeline steps without blocking the GUI.
"""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal


class StepThread(QThread):
    """
    StepThread UI helper class.
    """

    finished = Signal(object, str)

    def __init__(self, func: Callable[[], Any], parent: QObject | None = None) -> None:
        """
        Run   init  .

        :param func: See caller/context.
        :param parent: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__(func, parent)
            Out: UI/application state updated as intended.
        """
        super().__init__(parent)
        self._func = func

    def run(self) -> None:
        """
        Run run.

        :return: Result of this step or updated UI/application state.

        Example::
            In: run()
            Out: UI/application state updated as intended.
        """
        try:
            result = self._func()
            self.finished.emit(result, "")
        except Exception as error:
            self.finished.emit(None, str(error))
