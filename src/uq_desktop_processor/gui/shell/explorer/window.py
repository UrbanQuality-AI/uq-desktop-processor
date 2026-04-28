"""
Main explorer window: layout, menus, and coordination of map and tools.
"""

import logging

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
)

from uq_desktop_processor.gui.paths import APP_ICON_PATH
from uq_desktop_processor.gui.shell.constants import EULER_SOURCE_FIELD_LABELS, PLANNER_SOURCE_FIELD_LABELS
from uq_desktop_processor.gui.shell.explorer.console import ConsoleMixin
from uq_desktop_processor.gui.shell.explorer.dialogs import DialogsMixin
from uq_desktop_processor.gui.shell.explorer.modules import ModulesMixin
from uq_desktop_processor.gui.shell.explorer.pipeline import PipelineMixin
from uq_desktop_processor.gui.shell.explorer.sources import SourceInputsMixin
from uq_desktop_processor.gui.shell.logging_bridge import LogBridge, QtLogHandler
from uq_desktop_processor.gui.shell.ui.main_frame import build_main_frame
from uq_desktop_processor.gui.shell.worker import StepThread
from uq_desktop_processor.gui.styles.theme import NEON_STYLE
from uq_desktop_processor.gui.widgets import CompactStackedWidget, DeckMapWidget, NeonPanel
from uq_desktop_processor.pipeline import UrbanQualityAIPipeline
from uq_desktop_processor.street_view_analysis import EulerRoutesResult


class UrbanQualityAIExplorer(
    QMainWindow,
    ConsoleMixin,
    DialogsMixin,
    SourceInputsMixin,
    PipelineMixin,
    ModulesMixin,
):
    """Main window; shell UI is built by ``urban_quality_ai.gui.shell.ui``."""

    sidebar: QFrame
    button_group: QButtonGroup
    main_splitter: QSplitter
    tool_container: QFrame
    tool_layout: QVBoxLayout
    stacked_tools: QStackedWidget
    map_container: QFrame
    map_canvas: DeckMapWidget
    right_panel: QFrame
    ai_vision_box: NeonPanel
    metrics_text: QTextEdit
    logs_box: NeonPanel
    console: QTextEdit
    progress_bar: QProgressBar
    route_source_combo: QComboBox
    _route_input_stack: CompactStackedWidget
    route_city_edit: QLineEdit
    route_region_path_edit: QLineEdit
    route_roads_path_edit: QLineEdit
    _route_input_field_label: QLabel
    route_grid_cols: QSpinBox
    route_grid_rows: QSpinBox
    route_out_edit: QLineEdit
    route_consolidate_tol: QDoubleSpinBox
    route_cache_check: QCheckBox
    planner_source_combo: QComboBox
    _planner_input_stack: CompactStackedWidget
    place_name_edit: QLineEdit
    region_path_edit: QLineEdit
    road_path_edit: QLineEdit
    _planner_input_field_label: QLabel
    spacing_edit: QLineEdit
    min_dist_edit: QLineEdit
    planner_points_out_edit: QLineEdit
    token_edit: QLineEdit
    mapillary_points_edit: QLineEdit
    mapillary_out_edit: QLineEdit
    radius_edit: QLineEdit
    workers_spin: QSpinBox
    prefilter_image_folder_edit: QLineEdit
    prefilter_rejected_edit: QLineEdit
    model_combo: QComboBox
    device_combo: QComboBox
    threshold_slider: QSlider
    threshold_label: QLabel
    beta_spin: QDoubleSpinBox
    prefilter_pos_edit: QPlainTextEdit
    prefilter_neg_edit: QPlainTextEdit
    vit_images_dir_edit: QLineEdit
    vit_model_path_edit: QLineEdit
    vit_calibrators_edit: QLineEdit
    vit_output_path_edit: QLineEdit
    vit_output_layer_name_edit: QLineEdit
    vit_model_name_edit: QLineEdit
    vit_image_size_spin: QSpinBox
    vit_batch_size_spin: QSpinBox
    vit_device_combo: QComboBox
    run_buttons: list[QPushButton]

    def __init__(self) -> None:
        """
        Run   init  .

        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__()
            Out: UI/application state updated as intended.
        """
        super().__init__()
        self.setWindowTitle("UrbanQuality-AI 2026")
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1400, 850)
        self.setStyleSheet(NEON_STYLE)

        self.pipeline = UrbanQualityAIPipeline()
        self.current_active_module = -1
        self._worker: StepThread | None = None
        self.run_buttons = []
        self._last_euler_result: EulerRoutesResult | None = None

        self._log_bridge = LogBridge()
        self._log_bridge.append_text.connect(self._append_console)
        self._qt_log_handler = QtLogHandler(self._log_bridge)
        self._qt_log_handler.setLevel(logging.INFO)
        self._qt_log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(self._qt_log_handler)

        build_main_frame(self)

    def on_route_source_changed(self) -> None:
        """
        Run on route source changed.

        :return: Result of this step or updated UI/application state.

        Example::
            In: on_route_source_changed()
            Out: UI/application state updated as intended.
        """
        self._apply_source_input_page(
            self.route_source_combo,
            self._route_input_stack,
            self._route_input_field_label,
            EULER_SOURCE_FIELD_LABELS,
        )

    def on_planner_source_changed(self) -> None:
        """
        Run on planner source changed.

        :return: Result of this step or updated UI/application state.

        Example::
            In: on_planner_source_changed()
            Out: UI/application state updated as intended.
        """
        self._apply_source_input_page(
            self.planner_source_combo,
            self._planner_input_stack,
            self._planner_input_field_label,
            PLANNER_SOURCE_FIELD_LABELS,
        )
