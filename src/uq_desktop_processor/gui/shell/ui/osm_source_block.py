"""
UI block for selecting and validating OSM / road-graph input sources.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from uq_desktop_processor.gui.widgets import CompactStackedWidget

GEOJSON_FILTER = "GeoJSON (*.geojson *.json)"


@dataclass(frozen=True)
class OsmTripleSourceWidgets:
    """
    OsmTripleSourceWidgets UI helper class.
    """

    combo: QComboBox
    stack: CompactStackedWidget
    field_label: QLabel
    place_edit: QLineEdit
    region_edit: QLineEdit
    roads_edit: QLineEdit


def build_osm_triple_source_block(
    pick_file: Callable[[QLineEdit, str], None],
    field_labels: dict[str, str],
    on_changed: Callable[[], None],
    initial_place: str,
    *,
    roads_item_label: str,
) -> OsmTripleSourceWidgets:
    """
    Run build osm triple source block.

    :param pick_file: See caller/context.
    :param field_labels: See caller/context.
    :param on_changed: See caller/context.
    :param initial_place: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: build_osm_triple_source_block(pick_file, field_labels, on_changed, initial_place)
        Out: UI/application state updated as intended.
    """
    combo = QComboBox()
    combo.addItem("Place name (OSM)", "place")
    combo.addItem("Region polygon (GeoJSON)", "region")
    combo.addItem(roads_item_label, "roads")
    combo.currentIndexChanged.connect(on_changed)

    stack = CompactStackedWidget()

    page_place = QWidget()
    lay_place = QHBoxLayout(page_place)
    lay_place.setContentsMargins(0, 0, 0, 0)
    place_edit = QLineEdit(initial_place)
    lay_place.addWidget(place_edit)
    stack.addWidget(page_place)

    page_region = QWidget()
    lay_region = QHBoxLayout(page_region)
    lay_region.setContentsMargins(0, 0, 0, 0)
    region_edit = QLineEdit()
    region_edit.setPlaceholderText("polygon.geojson …")
    btn_region = QPushButton("…")
    btn_region.setFixedWidth(36)
    btn_region.clicked.connect(lambda: pick_file(region_edit, GEOJSON_FILTER))
    lay_region.addWidget(region_edit)
    lay_region.addWidget(btn_region)
    stack.addWidget(page_region)

    page_roads = QWidget()
    lay_roads = QHBoxLayout(page_roads)
    lay_roads.setContentsMargins(0, 0, 0, 0)
    roads_edit = QLineEdit()
    roads_edit.setPlaceholderText("roads.geojson …")
    btn_roads = QPushButton("…")
    btn_roads.setFixedWidth(36)
    btn_roads.clicked.connect(lambda: pick_file(roads_edit, GEOJSON_FILTER))
    lay_roads.addWidget(roads_edit)
    lay_roads.addWidget(btn_roads)
    stack.addWidget(page_roads)

    field_label = QLabel(field_labels["place"])
    return OsmTripleSourceWidgets(combo, stack, field_label, place_edit, region_edit, roads_edit)
