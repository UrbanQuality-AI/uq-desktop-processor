"""
Stacked tool page: generate and visualize road sampling points (GeoJSON output).
"""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from uq_desktop_processor.gui.shell.constants import PLANNER_SOURCE_FIELD_LABELS
from uq_desktop_processor.gui.shell.ui.osm_source_block import build_osm_triple_source_block
from uq_desktop_processor.gui.widgets import NeonPanel

if TYPE_CHECKING:
    from uq_desktop_processor.gui.shell.explorer import UrbanQualityAIExplorer


def add_sampling_points_page(explorer: "UrbanQualityAIExplorer") -> None:
    """
    Run add sampling points page.

    :param explorer: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: add_sampling_points_page(explorer)
        Out: UI/application state updated as intended.
    """
    p1 = QWidget()
    l1 = QVBoxLayout(p1)
    pan1 = NeonPanel(
        "Sampling points",
        "Generate a point layer along roads for image sampling.",
    )
    f1 = QFormLayout()

    planner = build_osm_triple_source_block(
        explorer.pick_file,
        PLANNER_SOURCE_FIELD_LABELS,
        explorer.on_planner_source_changed,
        "Katowice, Poland",
        roads_item_label="Road network file (GeoJSON)",
    )
    explorer.planner_source_combo = planner.combo
    explorer._planner_input_stack = planner.stack
    explorer.place_name_edit = planner.place_edit
    explorer.region_path_edit = planner.region_edit
    explorer.road_path_edit = planner.roads_edit
    explorer._planner_input_field_label = planner.field_label
    f1.addRow("Data source", explorer.planner_source_combo)
    f1.addRow(explorer._planner_input_field_label, explorer._planner_input_stack)

    explorer.spacing_edit = QLineEdit("100")
    f1.addRow("Distance between points (m)", explorer.spacing_edit)
    explorer.min_dist_edit = QLineEdit("50")
    f1.addRow("Min distance (m)", explorer.min_dist_edit)
    points_out_row = QHBoxLayout()
    explorer.planner_points_out_edit = QLineEdit("data/results/sampling_points.geojson")
    explorer.planner_points_out_edit.setPlaceholderText("path and filename (.geojson / .gpkg)")
    btn_points_out = QPushButton("…")
    btn_points_out.setFixedWidth(36)
    btn_points_out.clicked.connect(lambda: explorer.pick_save_points_layer(explorer.planner_points_out_edit))
    points_out_row.addWidget(explorer.planner_points_out_edit)
    points_out_row.addWidget(btn_points_out)
    f1.addRow("Output: point layer", points_out_row)

    pan1.main_layout.addLayout(f1)
    pan1.main_layout.addStretch()
    btn1 = QPushButton("GENERATE POINTS ▶")
    btn1.setObjectName("RunBtn")
    btn1.clicked.connect(explorer.on_run_planner)
    pan1.main_layout.addWidget(btn1)
    l1.addWidget(pan1)
    explorer.stacked_tools.addWidget(p1)
    explorer.run_buttons.append(btn1)
    explorer.on_planner_source_changed()
