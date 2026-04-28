"""
Composite PyDeck map widget: web view, layer panel, and redraw orchestration.
"""

from typing import Any

import geopandas as gpd
import pydeck as pdk
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from uq_desktop_processor.gui.map_view.constants import (
    MAP_LAYER_IDS_TOP_FIRST,
    MAP_LAYER_LABELS,
    MAP_REDRAW_DEBOUNCE_MS,
)
from uq_desktop_processor.gui.map_view.data import evaluation_scatter_data, fit_view_state
from uq_desktop_processor.gui.widgets.deck_map.layer_panel import LayerPanel
from uq_desktop_processor.gui.widgets.deck_map.layers import bounds_for_all_data, build_deck_layers, pick_tooltip
from uq_desktop_processor.gui.widgets.deck_map.web import DeckWebController
from uq_desktop_processor.gui.widgets.map_web_stack import MapWebStack


class DeckMapWidget(QWidget):
    """
    DeckMapWidget UI helper class.
    """

    def __init__(self) -> None:
        """
        Run   init  .

        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__()
            Out: UI/application state updated as intended.
        """
        super().__init__()
        self._planner_roads: gpd.GeoDataFrame | None = None
        self._planner_points: list[tuple[float, float]] = []
        self._euler_polylines: list[list[tuple[float, float]]] = []
        self._eval_results: dict[str, Any] | None = None

        self._stored_view: Any = None
        self._layers_first_auto_check: set[str] = set()

        self._map_redraw_timer = QTimer(self)
        self._map_redraw_timer.setSingleShot(True)
        self._map_redraw_timer.setInterval(MAP_REDRAW_DEBOUNCE_MS)
        self._map_redraw_timer.timeout.connect(self._flush_map_redraw)
        self._map_redraw_wants_refit = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._web = QWebEngineView(self)
        self._web.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        wes = self._web.settings()
        wes.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        wes.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self._web.setStyleSheet("background-color: #0d0d0d;")
        self._web.page().setBackgroundColor(QColor(0x0D, 0x0D, 0x0D))

        self._deck_web = DeckWebController(self._web)

        self._map_idle_overlay = QFrame(self)
        self._map_idle_overlay.setObjectName("MapIdleOverlay")
        self._map_idle_overlay.setStyleSheet(
            "QFrame#MapIdleOverlay { background-color: rgba(13, 13, 13, 0.96); border: 1px solid #2a2a2a; }"
        )
        idle_layout = QVBoxLayout(self._map_idle_overlay)
        idle_layout.addStretch(1)
        idle_msg = QLabel(
            "The map becomes active after generating the first preview layer\n"
            "(roads, sampling points, Euler/GPX routes, or CLIP results)."
        )
        idle_msg.setWordWrap(True)
        idle_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        idle_msg.setStyleSheet("color: #777; font-size: 12px; padding: 28px;")
        idle_layout.addWidget(idle_msg)
        idle_layout.addStretch(1)

        self._layer_panel = LayerPanel(
            self,
            layer_ids_top_first=list(MAP_LAYER_IDS_TOP_FIRST),
            layer_labels=MAP_LAYER_LABELS,
            on_changed=lambda: self._request_map_redraw(refit=False),
            on_reordered=lambda: self._request_map_redraw(refit=False),
            on_move_up=self._move_layer_up,
            on_move_down=self._move_layer_down,
        )

        self._map_web_host = MapWebStack(self._web, self._map_idle_overlay, self._layer_panel.widget)
        self._map_web_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._map_web_host, 1)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._sync_layer_list_items()
        self._apply_map_redraw(refit=True)

    def _has_roads_data(self) -> bool:
        """
        Run  has roads data.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _has_roads_data()
            Out: UI/application state updated as intended.
        """
        return self._planner_roads is not None and not self._planner_roads.empty

    def _has_points_data(self) -> bool:
        """
        Run  has points data.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _has_points_data()
            Out: UI/application state updated as intended.
        """
        return bool(self._planner_points)

    def _has_euler_data(self) -> bool:
        """
        Run  has euler data.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _has_euler_data()
            Out: UI/application state updated as intended.
        """
        return any(len(polyline) >= 2 for polyline in self._euler_polylines)

    def _has_clip_data(self) -> bool:
        """
        Run  has clip data.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _has_clip_data()
            Out: UI/application state updated as intended.
        """
        if not self._eval_results:
            return False
        lon_values, _lat_values, _score_values = evaluation_scatter_data(self._eval_results)
        return lon_values.size > 0

    def _layer_has_data(self, layer_id: str) -> bool:
        """
        Run  layer has data.

        :param layer_id: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _layer_has_data(layer_id)
            Out: UI/application state updated as intended.
        """
        return {
            "roads": self._has_roads_data(),
            "points": self._has_points_data(),
            "euler": self._has_euler_data(),
            "clip": self._has_clip_data(),
        }[layer_id]

    def _has_any_map_data(self) -> bool:
        """
        Run  has any map data.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _has_any_map_data()
            Out: UI/application state updated as intended.
        """
        return any(self._layer_has_data(lid) for lid in MAP_LAYER_IDS_TOP_FIRST)

    def _sync_layer_list_items(self) -> None:
        """
        Run  sync layer list items.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _sync_layer_list_items()
            Out: UI/application state updated as intended.
        """
        self._layer_panel.list.blockSignals(True)
        try:
            for layer_id in MAP_LAYER_IDS_TOP_FIRST:
                item = self._layer_panel.item_for_layer(layer_id)
                if item is None:
                    continue
                has_data = self._layer_has_data(layer_id)
                if has_data:
                    item.setFlags(self._layer_panel.active_item_flags())
                    if layer_id not in self._layers_first_auto_check:
                        # Auto-enable a layer once when its data appears for the first time.
                        item.setCheckState(Qt.CheckState.Checked)
                        self._layers_first_auto_check.add(layer_id)
                else:
                    # Data disappeared (or was reset), so disable the item again.
                    self._layers_first_auto_check.discard(layer_id)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setFlags(self._layer_panel.inactive_item_flags())
        finally:
            self._layer_panel.list.blockSignals(False)

    def _move_layer_up(self) -> None:
        """
        Run  move layer up.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _move_layer_up()
            Out: UI/application state updated as intended.
        """
        row = self._layer_panel.list.currentRow()
        if row <= 0:
            return
        self._layer_panel.list.blockSignals(True)
        try:
            item = self._layer_panel.list.takeItem(row)
            if item:
                self._layer_panel.list.insertItem(row - 1, item)
                self._layer_panel.list.setCurrentRow(row - 1)
        finally:
            self._layer_panel.list.blockSignals(False)
        self._request_map_redraw(refit=False)

    def _move_layer_down(self) -> None:
        """
        Run  move layer down.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _move_layer_down()
            Out: UI/application state updated as intended.
        """
        row = self._layer_panel.list.currentRow()
        # Nothing to move when no selection or already at the last row.
        if row < 0 or row >= self._layer_panel.list.count() - 1:
            return
        self._layer_panel.list.blockSignals(True)
        try:
            item = self._layer_panel.list.takeItem(row)
            if item:
                self._layer_panel.list.insertItem(row + 1, item)
                self._layer_panel.list.setCurrentRow(row + 1)
        finally:
            self._layer_panel.list.blockSignals(False)
        self._request_map_redraw(refit=False)

    def _bounds_for_all_data(self) -> tuple[list[float], list[float]]:
        """
        Run  bounds for all data.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _bounds_for_all_data()
            Out: UI/application state updated as intended.
        """
        return bounds_for_all_data(
            planner_roads=self._planner_roads,
            planner_points=self._planner_points,
            euler_polylines=self._euler_polylines,
            eval_results=self._eval_results,
        )

    def _build_deck_layers(self) -> list[Any]:
        """
        Run  build deck layers.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _build_deck_layers()
            Out: UI/application state updated as intended.
        """
        order_top_first = self._layer_panel.layer_order_top_to_bottom()
        checked = {layer_id: self._layer_panel.layer_checked(layer_id) for layer_id in MAP_LAYER_IDS_TOP_FIRST}
        has_data = {layer_id: self._layer_has_data(layer_id) for layer_id in MAP_LAYER_IDS_TOP_FIRST}
        return build_deck_layers(
            layer_order_top_to_bottom=order_top_first,
            layer_checked=checked,
            layer_has_data=has_data,
            planner_roads=self._planner_roads,
            planner_points=self._planner_points,
            euler_polylines=self._euler_polylines,
            eval_results=self._eval_results,
        )

    def _pick_tooltip(self) -> dict[str, Any]:
        """
        Run  pick tooltip.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _pick_tooltip()
            Out: UI/application state updated as intended.
        """
        checked = {layer_id: self._layer_panel.layer_checked(layer_id) for layer_id in MAP_LAYER_IDS_TOP_FIRST}
        has_data = {layer_id: self._layer_has_data(layer_id) for layer_id in MAP_LAYER_IDS_TOP_FIRST}
        return pick_tooltip(layer_checked=checked, layer_has_data=has_data)

    def _request_map_redraw(self, refit: bool) -> None:
        """
        Run  request map redraw.

        :param refit: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _request_map_redraw(refit)
            Out: UI/application state updated as intended.
        """
        self._map_redraw_wants_refit = self._map_redraw_wants_refit or refit
        self._map_redraw_timer.start()

    def _flush_map_redraw(self) -> None:
        """
        Run  flush map redraw.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _flush_map_redraw()
            Out: UI/application state updated as intended.
        """
        refit = self._map_redraw_wants_refit
        self._map_redraw_wants_refit = False
        self._apply_map_redraw(refit=refit)

    def _apply_map_redraw(self, *, refit: bool) -> None:
        """
        Run  apply map redraw.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _apply_map_redraw()
            Out: UI/application state updated as intended.
        """
        if not self._has_any_map_data():
            self._deck_web.clear_pending()
            self._stored_view = None
            self._map_idle_overlay.setVisible(True)
            self._web.setEnabled(False)
            self._web.setHtml(
                '<!DOCTYPE html><html><head><meta charset="utf-8"/>'
                "<style>html,body{margin:0;height:100%;background:#0d0d0d;}</style></head><body></body></html>",
                QUrl("about:blank"),
            )
            return

        self._map_idle_overlay.setVisible(False)
        self._web.setEnabled(True)

        longitudes, latitudes = self._bounds_for_all_data()

        if refit or self._stored_view is None or not self._deck_web.deck_surface_ready:
            self._deck_web.map_commit_generation += 1
            self._stored_view = fit_view_state(longitudes, latitudes)
            deck = pdk.Deck(
                layers=self._build_deck_layers(),
                initial_view_state=self._stored_view,
                map_style=pdk.map_styles.CARTO_DARK,
                tooltip=self._pick_tooltip(),
            )
            self._deck_web.set_deck_page_html(deck)
            return

        def _on_view(view: pdk.ViewState | None, generation: int) -> None:
            """
            Run  on view.

            :param view: See caller/context.
            :param generation: See caller/context.
            :return: Result of this step or updated UI/application state.

            Example::
                In: _on_view(view, generation)
                Out: UI/application state updated as intended.
            """
            if generation != self._deck_web.map_commit_generation:
                return
            if view is not None:
                self._stored_view = view
            elif self._stored_view is None:
                self._stored_view = fit_view_state(longitudes, latitudes)
            deck = pdk.Deck(
                layers=self._build_deck_layers(),
                initial_view_state=self._stored_view,
                map_style=pdk.map_styles.CARTO_DARK,
                tooltip=self._pick_tooltip(),
            )
            self._deck_web.set_deck_page_html(deck)

        self._deck_web.schedule_reload_preserving_view(on_view=_on_view)

    def update_planner_layers(
        self,
        roads_gdf: gpd.GeoDataFrame | None,
        points: list[tuple[float, float]] | None,
    ) -> None:
        """
        Run update planner layers.

        :param roads_gdf: See caller/context.
        :param points: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: update_planner_layers(roads_gdf, points)
            Out: UI/application state updated as intended.
        """
        self._planner_roads = roads_gdf
        self._planner_points = list(points) if points else []
        self._sync_layer_list_items()
        self._request_map_redraw(refit=True)

    def update_euler_layer(self, polylines: list[list[tuple[float, float]]]) -> None:
        """
        Run update euler layer.

        :param polylines: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: update_euler_layer(polylines)
            Out: UI/application state updated as intended.
        """
        self._euler_polylines = [list(polyline) for polyline in polylines]
        self._sync_layer_list_items()
        self._request_map_redraw(refit=True)

    def update_eval_layer(self, _roads_gdf: gpd.GeoDataFrame | None, results: dict[str, Any] | None) -> None:
        """
        Run update eval layer.

        :param _roads_gdf: See caller/context.
        :param results: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: update_eval_layer(_roads_gdf, results)
            Out: UI/application state updated as intended.
        """
        if results:
            lon_values, _lat_values, _score_values = evaluation_scatter_data(results)
            self._eval_results = results if lon_values.size != 0 else None
        else:
            self._eval_results = None
        self._sync_layer_list_items()
        self._request_map_redraw(refit=True)
