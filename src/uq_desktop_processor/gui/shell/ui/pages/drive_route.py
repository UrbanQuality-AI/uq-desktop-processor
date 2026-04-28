"""
Stacked tool page: Chinese postman routes and GPX export for drive coverage.
"""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from uq_desktop_processor.gui.shell.constants import EULER_SOURCE_FIELD_LABELS
from uq_desktop_processor.gui.shell.ui.osm_source_block import build_osm_triple_source_block
from uq_desktop_processor.gui.widgets import NeonPanel

if TYPE_CHECKING:
    from uq_desktop_processor.gui.shell.explorer import UrbanQualityAIExplorer


def add_drive_route_page(explorer: "UrbanQualityAIExplorer") -> None:
    """
    Run add drive route page.

    :param explorer: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: add_drive_route_page(explorer)
        Out: UI/application state updated as intended.
    """
    p0 = QWidget()
    l0 = QVBoxLayout(p0)
    pan0 = NeonPanel(
        "Drive route (Chinese postman/GPX)",
        "Generate a drivable GPX route over the road network.",
    )
    f0 = QFormLayout()

    euler = build_osm_triple_source_block(
        explorer.pick_file,
        EULER_SOURCE_FIELD_LABELS,
        explorer.on_route_source_changed,
        "Katowice, Poland",
        roads_item_label="Road network (GeoJSON)",
    )
    explorer.route_source_combo = euler.combo
    explorer._route_input_stack = euler.stack
    explorer.route_city_edit = euler.place_edit
    explorer.route_region_path_edit = euler.region_edit
    explorer.route_roads_path_edit = euler.roads_edit
    explorer._route_input_field_label = euler.field_label
    f0.addRow("Data source", explorer.route_source_combo)
    f0.addRow(explorer._route_input_field_label, explorer._route_input_stack)

    explorer.route_grid_cols = QSpinBox()
    explorer.route_grid_cols.setRange(1, 30)
    explorer.route_grid_cols.setValue(3)
    f0.addRow("Grid columns", explorer.route_grid_cols)
    explorer.route_grid_rows = QSpinBox()
    explorer.route_grid_rows.setRange(1, 30)
    explorer.route_grid_rows.setValue(3)
    f0.addRow("Grid rows", explorer.route_grid_rows)

    out_row = QHBoxLayout()
    explorer.route_out_edit = QLineEdit("routes_chinese_postman_gpx")
    explorer.route_out_edit.setPlaceholderText("Output folder for .gpx files")
    btn_out = QPushButton("…")
    btn_out.setFixedWidth(36)
    btn_out.clicked.connect(lambda: explorer.pick_folder(explorer.route_out_edit))
    out_row.addWidget(explorer.route_out_edit)
    out_row.addWidget(btn_out)
    f0.addRow("Output folder", out_row)

    explorer.route_consolidate_tol = QDoubleSpinBox()
    explorer.route_consolidate_tol.setRange(5.0, 80.0)
    explorer.route_consolidate_tol.setValue(15.0)
    explorer.route_consolidate_tol.setSuffix(" m")
    f0.addRow("Consolidate tolerance", explorer.route_consolidate_tol)
    explorer.route_cache_check = QCheckBox("Use OSMnx cache")
    explorer.route_cache_check.setChecked(True)
    f0.addRow("", explorer.route_cache_check)

    pan0.main_layout.addLayout(f0)
    pan0.main_layout.addStretch()
    btn0 = QPushButton("GENERATE ROUTE ▶")
    btn0.setObjectName("RunBtn")
    btn0.clicked.connect(explorer.on_run_euler_routes)
    pan0.main_layout.addWidget(btn0)
    l0.addWidget(pan0)
    explorer.stacked_tools.addWidget(p0)
    explorer.run_buttons.append(btn0)
    explorer.on_route_source_changed()
