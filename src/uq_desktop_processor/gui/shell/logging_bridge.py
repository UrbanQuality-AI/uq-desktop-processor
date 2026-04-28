"""
Forwards Python logging records into Qt text widgets for in-app log panes.
"""

import logging

from PySide6.QtCore import QObject, Signal


class LogBridge(QObject):
    """
    LogBridge UI helper class.
    """

    append_text = Signal(str)


class QtLogHandler(logging.Handler):
    """
    QtLogHandler UI helper class.
    """

    def __init__(self, bridge: LogBridge) -> None:
        """
        Run   init  .

        :param bridge: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__(bridge)
            Out: UI/application state updated as intended.
        """
        super().__init__()
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        """
        Run emit.

        :param record: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: emit(record)
            Out: UI/application state updated as intended.
        """
        try:
            formatted_message = self.format(record)
            self._bridge.append_text.emit(formatted_message)
        except (KeyboardInterrupt, SystemExit):
            raise
        except (RuntimeError, ValueError, TypeError, AttributeError):
            self.handleError(record)
