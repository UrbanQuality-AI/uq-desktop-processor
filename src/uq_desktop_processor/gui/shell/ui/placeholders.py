"""
Placeholder pages and stub widgets for shell sections not yet implemented.
"""

from PySide6.QtWidgets import QPushButton, QStackedWidget, QVBoxLayout, QWidget

from uq_desktop_processor.gui.widgets import NeonPanel


def add_placeholder_tool_page(
    stacked_tools: QStackedWidget,
    run_buttons: list[QPushButton],
    *,
    title: str,
    description: str,
) -> None:
    """
    Run add placeholder tool page.

    :param stacked_tools: See caller/context.
    :param run_buttons: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: add_placeholder_tool_page(stacked_tools, run_buttons)
        Out: UI/application state updated as intended.
    """
    page = QWidget()
    layout = QVBoxLayout(page)
    panel = NeonPanel(title, description)
    panel.main_layout.addStretch()
    btn = QPushButton("COMING SOON")
    btn.setObjectName("RunBtn")
    btn.setEnabled(False)
    panel.main_layout.addWidget(btn)
    layout.addWidget(panel)
    stacked_tools.addWidget(page)
    run_buttons.append(btn)
