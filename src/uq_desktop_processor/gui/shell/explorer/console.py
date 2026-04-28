"""
Embedded console widget for the explorer window (command/output surface).
"""

from typing import Protocol

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextEdit


class ConsoleSupported(Protocol):
    """
    ConsoleSupported UI helper class.
    """

    console: QTextEdit
    metrics_text: QTextEdit


class ConsoleMixin:
    """
    ConsoleMixin UI helper class.
    """

    def _append_console(self: ConsoleSupported, line: str) -> None:
        """
        Run  append console.

        :param line: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _append_console(line)
            Out: UI/application state updated as intended.
        """
        self.console.append(line.rstrip())

    def _append_data_science_section(self: ConsoleSupported, title: str, lines: list[str]) -> None:
        """
        Run  append data science section.

        :param title: See caller/context.
        :param lines: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _append_data_science_section(title, lines)
            Out: UI/application state updated as intended.
        """
        current = self.metrics_text.toPlainText().rstrip()
        sep = "\n\n" + "─" * 44 + "\n\n" if current else ""
        body = "\n".join(lines)
        self.metrics_text.setPlainText(f"{current}{sep}{title}\n{body}")
        self.metrics_text.moveCursor(QTextCursor.MoveOperation.End)
