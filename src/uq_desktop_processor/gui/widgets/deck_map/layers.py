"""
Constructs PyDeck layers from geodata and evaluation overlays with tooltips.
"""

import json
import math
from typing import Any

import geopandas as gpd
import pydeck as pdk

from uq_desktop_processor.gui.map_view.constants import EULER_LINE_COLORS
from uq_desktop_processor.gui.map_view.data import (
    euler_polylines_for_display,
    evaluation_scatter_data,
    map_display_budgets,
    red_yellow_green_rgba,
    simplify_roads_gdf_for_map_display,
)


def roads_geojson_layer(
    roads_gdf: gpd.GeoDataFrame | None,
    *,
    simplify_meters: float,
    max_features: int,
    line_rgba: tuple[int, int, int, int] = (68, 68, 68, 235),
) -> Any:
    """
    Run roads geojson layer.

    :param roads_gdf: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: roads_geojson_layer(roads_gdf)
        Out: UI/application state updated as intended.
    """
    if roads_gdf is None or roads_gdf.empty:
        return None
    display_roads_gdf = simplify_roads_gdf_for_map_display(
        roads_gdf.to_crs(4326),
        simplify_meters=simplify_meters,
        max_features=max_features,
    )
    if display_roads_gdf.empty:
        return None
    data = json.loads(display_roads_gdf.to_json())
    return pdk.Layer(
        "GeoJsonLayer",
        id="roads-layer",
        data=data,
        stroked=True,
        filled=False,
        line_width_min_pixels=1,
        get_line_color=list(line_rgba),
        pickable=False,
        opacity=0.8,
        parameters={"depthTest": False},
    )


def append_bounds_roads(roads_gdf: gpd.GeoDataFrame | None, lons: list[float], lats: list[float]) -> None:
    """
    Run append bounds roads.

    :param roads_gdf: See caller/context.
    :param lons: See caller/context.
    :param lats: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: append_bounds_roads(roads_gdf, lons, lats)
        Out: UI/application state updated as intended.
    """
    if roads_gdf is None or roads_gdf.empty:
        return
    bounds = roads_gdf.to_crs(4326).total_bounds
    lons.extend([float(bounds[0]), float(bounds[2])])
    lats.extend([float(bounds[1]), float(bounds[3])])


def bounds_for_all_data(
    *,
    planner_roads: gpd.GeoDataFrame | None,
    planner_points: list[tuple[float, float]],
    euler_polylines: list[list[tuple[float, float]]],
    eval_results: dict[str, Any] | None,
) -> tuple[list[float], list[float]]:
    """
    Run bounds for all data.

    :return: Result of this step or updated UI/application state.

    Example::
        In: bounds_for_all_data()
        Out: UI/application state updated as intended.
    """
    lons: list[float] = []
    lats: list[float] = []
    append_bounds_roads(planner_roads, lons, lats)
    for lat, lon in planner_points:
        lons.append(float(lon))
        lats.append(float(lat))
    for poly in euler_polylines:
        for lon, la in poly:
            lons.append(float(lon))
            lats.append(float(la))
    if eval_results:
        lon, la, _s = evaluation_scatter_data(eval_results)
        for i in range(len(lon)):
            lons.append(float(lon[i]))
            lats.append(float(la[i]))
    return lons, lats


def build_deck_layers(
    *,
    layer_order_top_to_bottom: list[str],
    layer_checked: dict[str, bool],
    layer_has_data: dict[str, bool],
    planner_roads: gpd.GeoDataFrame | None,
    planner_points: list[tuple[float, float]],
    euler_polylines: list[list[tuple[float, float]]],
    eval_results: dict[str, Any] | None,
) -> list[Any]:
    """
    Run build deck layers.

    :return: Result of this step or updated UI/application state.

    Example::
        In: build_deck_layers()
        Out: UI/application state updated as intended.
    """
    order_bottom_first = list(reversed(layer_order_top_to_bottom))
    layers: list[Any] = []
    visible = sum(1 for lid, on in layer_checked.items() if on and layer_has_data.get(lid, False))
    budgets = map_display_budgets(visible)
    common_params = {"depthTest": False}

    for lid in order_bottom_first:
        if not layer_checked.get(lid, False) or not layer_has_data.get(lid, False):
            continue
        if lid == "roads":
            layer = roads_geojson_layer(
                planner_roads,
                simplify_meters=float(budgets["road_simplify_meters"]),
                max_features=int(budgets["road_max_features"]),
            )
            if layer is not None:
                layers.append(layer)
        elif lid == "points":
            rows = [{"lon": float(lon), "lat": float(lat)} for lat, lon in planner_points]
            cap = int(budgets["points_max"])
            if len(rows) > cap:
                step = max(1, math.ceil(len(rows) / cap))
                rows = rows[::step]
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    id="points-layer",
                    data=rows,
                    get_position="[lon, lat]",
                    get_fill_color=[0, 242, 255, 195],
                    get_line_color=[0, 242, 255, 255],
                    stroked=True,
                    line_width_min_pixels=1,
                    get_radius=22,
                    radius_min_pixels=4,
                    radius_max_pixels=28,
                    pickable=True,
                    parameters=common_params,
                )
            )
        elif lid == "euler":
            paths: list[dict[str, Any]] = []
            euler_polys = euler_polylines_for_display(
                euler_polylines,
                int(budgets["euler_vertex_budget"]),
            )
            for polyline_index, polyline in enumerate(euler_polys):
                if len(polyline) < 2:
                    continue
                rgb = EULER_LINE_COLORS[polyline_index % len(EULER_LINE_COLORS)]
                path = [[float(lon), float(lat)] for lon, lat in polyline]
                paths.append({"path": path, "name": f"Sector {polyline_index + 1}", "color": [*rgb, 235]})
            if paths:
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        id="euler-layer",
                        data=paths,
                        get_path="path",
                        get_color="color",
                        width_min_pixels=3,
                        cap_rounded=True,
                        joint_rounded=True,
                        pickable=True,
                        parameters=common_params,
                    )
                )
        elif lid == "clip" and eval_results is not None:
            lon_values, lat_values, scores = evaluation_scatter_data(eval_results)
            if lon_values.size == 0:
                continue
            colors = red_yellow_green_rgba(scores)
            rows = [
                {
                    "lon": float(lon_values[row_index]),
                    "lat": float(lat_values[row_index]),
                    "overall": round(float(scores[row_index]), 1),
                    "color": colors[row_index],
                }
                for row_index in range(len(lon_values))
            ]
            cap = int(budgets["points_max"])
            if len(rows) > cap:
                step = max(1, math.ceil(len(rows) / cap))
                rows = rows[::step]
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    id="clip-layer",
                    data=rows,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_line_color=[0, 242, 255, 200],
                    stroked=True,
                    line_width_min_pixels=1,
                    get_radius=26,
                    radius_min_pixels=5,
                    radius_max_pixels=32,
                    pickable=True,
                    parameters=common_params,
                )
            )
    return layers


def pick_tooltip(*, layer_checked: dict[str, bool], layer_has_data: dict[str, bool]) -> dict[str, Any]:
    """
    Run pick tooltip.

    :return: Result of this step or updated UI/application state.

    Example::
        In: pick_tooltip()
        Out: UI/application state updated as intended.
    """
    pickable_on = sum(
        1
        for layer_id in ("clip", "euler", "points")
        if layer_checked.get(layer_id, False) and layer_has_data.get(layer_id, False)
    )
    if pickable_on > 1:
        return {
            "html": "<b>Layer:</b> {layer}<br/>Configure the view in the panel",
            "style": {"backgroundColor": "#111", "color": "#fff", "fontSize": "10px"},
        }
    if layer_checked.get("clip", False) and layer_has_data.get("clip", False):
        return {
            "html": "<b>Avg estimation</b><br/>{overall}%",
            "style": {"backgroundColor": "#111", "color": "#00ff41", "fontSize": "12px"},
        }
    if layer_checked.get("euler", False) and layer_has_data.get("euler", False):
        return {
            "html": "<b>{name}</b><br/>Euler / GPX",
            "style": {"backgroundColor": "#111", "color": "#00f2ff"},
        }
    if layer_checked.get("points", False) and layer_has_data.get("points", False):
        return {"html": "<b>Point</b><br/>{lat}, {lon}", "style": {"color": "#00f2ff"}}
    return {}
