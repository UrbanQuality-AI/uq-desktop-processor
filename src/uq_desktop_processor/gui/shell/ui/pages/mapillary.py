"""
Stacked tool page: Mapillary image search and download around sampling points.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from uq_desktop_processor.gui.widgets import NeonPanel

if TYPE_CHECKING:
    from uq_desktop_processor.gui.shell.explorer import UrbanQualityAIExplorer


def add_mapillary_page(explorer: "UrbanQualityAIExplorer") -> None:
    """
    Run add mapillary page.

    :param explorer: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: add_mapillary_page(explorer)
        Out: UI/application state updated as intended.
    """
    p2 = QWidget()
    l2 = QVBoxLayout(p2)
    pan2 = NeonPanel(
        "Mapillary download",
        "Download street-level images for the generated sampling points.",
    )
    f2 = QFormLayout()
    explorer.token_edit = QLineEdit()
    explorer.token_edit.setPlaceholderText("Paste token or set MAPILLARY_ACCESS_TOKEN")
    f2.addRow("Mapillary token", explorer.token_edit)

    pts_in_row = QHBoxLayout()
    explorer.mapillary_points_edit = QLineEdit(str(Path("data") / "results" / "sampling_points.geojson"))
    explorer.mapillary_points_edit.setPlaceholderText(".geojson / .gpkg from point generation step")
    btn_mapillary_pts = QPushButton("…")
    btn_mapillary_pts.setFixedWidth(36)
    btn_mapillary_pts.clicked.connect(
        lambda: explorer.pick_file(
            explorer.mapillary_points_edit,
            "GeoJSON (*.geojson);;GeoPackage (*.gpkg)",
        )
    )
    pts_in_row.addWidget(explorer.mapillary_points_edit)
    pts_in_row.addWidget(btn_mapillary_pts)
    f2.addRow("Input: point layer", pts_in_row)

    img_out_row = QHBoxLayout()
    explorer.mapillary_out_edit = QLineEdit(str(Path("data") / "images" / "raw"))
    explorer.mapillary_out_edit.setPlaceholderText("Folder for downloaded JPEGs (used by later CLIP steps)")
    btn_mapillary_out = QPushButton("…")
    btn_mapillary_out.setFixedWidth(36)
    btn_mapillary_out.clicked.connect(lambda: explorer.pick_folder(explorer.mapillary_out_edit))
    img_out_row.addWidget(explorer.mapillary_out_edit)
    img_out_row.addWidget(btn_mapillary_out)
    f2.addRow("Output: image folder", img_out_row)

    explorer.radius_edit = QLineEdit("150")
    f2.addRow("Search radius (m)", explorer.radius_edit)
    explorer.workers_spin = QSpinBox()
    explorer.workers_spin.setRange(1, 64)
    explorer.workers_spin.setValue(20)
    f2.addRow("Parallel workers", explorer.workers_spin)
    pan2.main_layout.addLayout(f2)
    pan2.main_layout.addStretch()
    btn2 = QPushButton("DOWNLOAD IMAGES ⬇")
    btn2.setObjectName("RunBtn")
    btn2.clicked.connect(explorer.on_run_download)
    pan2.main_layout.addWidget(btn2)
    l2.addWidget(pan2)
    explorer.stacked_tools.addWidget(p2)
    explorer.run_buttons.append(btn2)
