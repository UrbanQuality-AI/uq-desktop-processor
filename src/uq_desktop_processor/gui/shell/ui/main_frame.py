"""
Primary application frame hosting the stacked tool pages and central splitter.
"""

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from uq_desktop_processor.gui.shell.ui.module_pages import setup_module_pages
from uq_desktop_processor.gui.widgets import DeckMapWidget, NeonPanel

if TYPE_CHECKING:
    from uq_desktop_processor.gui.shell.explorer import UrbanQualityAIExplorer


def build_main_frame(explorer: "UrbanQualityAIExplorer") -> None:
    """
    Run build main frame.

    :param explorer: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: build_main_frame(explorer)
        Out: UI/application state updated as intended.
    """
    central_widget = QWidget()
    explorer.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    top_layout = QHBoxLayout()
    top_layout.setSpacing(0)
    top_layout.setContentsMargins(0, 0, 0, 0)

    explorer.sidebar = QFrame()
    explorer.sidebar.setObjectName("Sidebar")
    explorer.sidebar.setFixedWidth(70)
    sidebar_layout = QVBoxLayout(explorer.sidebar)
    sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    explorer.button_group = QButtonGroup(explorer)
    explorer.button_group.setExclusive(True)

    for panel_index, (icon, name) in enumerate(
        [
            ("1", "Drive route"),
            ("2", "Sampling points"),
            ("3", "Mapillary download"),
            ("4", "Panoramas → views"),
            ("5", "CLIP prefilter"),
            ("6", "ViT scoring"),
            ("7", "Export results"),
            ("8", "Validation"),
        ]
    ):
        sidebar_button = QPushButton(icon)
        sidebar_button.setObjectName("SidebarBtn")
        sidebar_button.setCheckable(True)
        sidebar_button.setToolTip(name)
        sidebar_button.setFixedSize(70, 70)
        sidebar_button.clicked.connect(
            lambda _checked=False, panel_idx=panel_index: explorer.toggle_module_panel(panel_idx)
        )
        explorer.button_group.addButton(sidebar_button, panel_index)
        sidebar_layout.addWidget(sidebar_button)

    top_layout.addWidget(explorer.sidebar)

    explorer.main_splitter = QSplitter(Qt.Orientation.Horizontal)
    explorer.main_splitter.setHandleWidth(6)

    explorer.tool_container = QFrame()
    explorer.tool_container.setObjectName("ToolPanelContainer")
    explorer.tool_container.setMinimumWidth(320)
    explorer.tool_layout = QVBoxLayout(explorer.tool_container)

    explorer.stacked_tools = QStackedWidget()
    explorer.tool_layout.addWidget(explorer.stacked_tools)

    setup_module_pages(explorer)
    explorer.tool_container.setVisible(False)

    explorer.map_container = QFrame()
    explorer.map_container.setObjectName("PanelBackground")
    map_layout = QVBoxLayout(explorer.map_container)
    map_header = QLabel("Live Map View")
    map_layout.addWidget(map_header)

    explorer.map_canvas = DeckMapWidget()
    map_layout.addWidget(explorer.map_canvas, 1)

    explorer.right_panel = QFrame()
    explorer.right_panel.setMinimumWidth(300)
    right_layout = QVBoxLayout(explorer.right_panel)

    explorer.ai_vision_box = NeonPanel("Data Science Console")
    explorer.metrics_text = QTextEdit()
    explorer.metrics_text.setReadOnly(True)
    explorer.metrics_text.setObjectName("Console")
    explorer.metrics_text.setMaximumHeight(160)
    explorer.metrics_text.setPlaceholderText(
        "Pipeline summaries will appear here. Use the tools on the left to run a step; each "
        "completed run appends metrics and output paths below."
    )
    explorer.ai_vision_box.main_layout.addWidget(explorer.metrics_text)
    explorer.ai_vision_box.main_layout.addStretch()
    right_layout.addWidget(explorer.ai_vision_box, 1)

    explorer.logs_box = NeonPanel("System Logs")
    explorer.console = QTextEdit()
    explorer.console.setObjectName("Console")
    explorer.console.setReadOnly(True)
    explorer.console.append("[INFO] System Ready...")
    explorer.logs_box.main_layout.addWidget(explorer.console)
    right_layout.addWidget(explorer.logs_box, 2)

    explorer.main_splitter.addWidget(explorer.tool_container)
    explorer.main_splitter.addWidget(explorer.map_container)
    explorer.main_splitter.addWidget(explorer.right_panel)

    explorer.main_splitter.setStretchFactor(1, 1)

    top_layout.addWidget(explorer.main_splitter)
    main_layout.addLayout(top_layout, 1)

    explorer.progress_bar = QProgressBar()
    main_layout.addWidget(explorer.progress_bar)
