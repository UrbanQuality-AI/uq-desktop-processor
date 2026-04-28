"""
Builds numpy/PyDeck-friendly data from GeoJSON, sampling points, and evaluation results.
"""

import math
from typing import Any

import geopandas as gpd
import numpy as np
import pydeck as pdk

from uq_desktop_processor.gui.map_view.constants import (
    EULER_VERTEX_BUDGET_BASE,
    POINTS_MAP_MAX_BASE,
    ROADS_MAP_MAX_FEATURES_BASE,
    ROADS_MAP_SIMPLIFY_METERS_BASE,
)
from uq_desktop_processor.layer_creation.vector_layers.point_layer.utils import parse_lat_lon


def evaluation_scatter_data(results: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run evaluation scatter data.

    :param results: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: evaluation_scatter_data(results)
        Out: UI/application state updated as intended.
    """
    xs: list[float] = []
    ys: list[float] = []
    scores: list[float] = []
    for image_result in results.get("images") or []:
        image_path = image_result.get("image_path") or ""
        try:
            if not image_path:
                raise ValueError("Missing image_path in evaluation record.")
            lat, lon = parse_lat_lon(image_path)
        except (ValueError, IndexError):
            continue
        xs.append(lon)
        ys.append(lat)
        scores.append(float(image_result.get("overall_pct") or 0.0))
    if not xs:
        return np.array([]), np.array([]), np.array([])
    return np.array(xs), np.array(ys), np.array(scores)


def fit_view_state(lons: list[float], lats: list[float]) -> pdk.ViewState:
    """
    Run fit view state.

    :param lons: See caller/context.
    :param lats: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: fit_view_state(lons, lats)
        Out: UI/application state updated as intended.
    """
    if not lons or not lats:
        return pdk.ViewState(latitude=50.0, longitude=19.0, zoom=4, pitch=0, bearing=0)
    lo, hi = min(lons), max(lons)
    la_lo, la_hi = min(lats), max(lats)
    span = max(hi - lo, la_hi - la_lo, 1e-6)
    pad = 0.08 * span
    lo, hi = lo - pad, hi + pad
    la_lo, la_hi = la_lo - pad, la_hi + pad
    center_lon = (lo + hi) / 2
    center_lat = (la_lo + la_hi) / 2
    lon_span = max(hi - lo, 1e-6)
    zoom = math.log2(360 / lon_span) - 0.75
    zoom = max(2.0, min(17.0, zoom))
    return pdk.ViewState(
        latitude=float(center_lat),
        longitude=float(center_lon),
        zoom=float(zoom),
        pitch=0,
        bearing=0,
    )


def red_yellow_green_rgba(scores: np.ndarray) -> list[list[int]]:
    """
    Run red yellow green rgba.

    :param scores: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: red_yellow_green_rgba(scores)
        Out: UI/application state updated as intended.
    """
    if scores.size == 0:
        return []
    vmin, vmax = float(np.min(scores)), float(np.max(scores))
    if vmax <= vmin:
        vmin -= 1.0
        vmax += 1.0
    colors_rgba: list[list[int]] = []
    for score_value in scores:
        normalized_score = (float(score_value) - vmin) / (vmax - vmin)
        normalized_score = max(0.0, min(1.0, normalized_score))
        if normalized_score <= 0.5:
            blend_factor = normalized_score / 0.5
            red_channel, green_channel, blue_channel = 255, int(255 * blend_factor), 0
        else:
            blend_factor = (normalized_score - 0.5) / 0.5
            red_channel, green_channel, blue_channel = int(255 * (1.0 - blend_factor)), 255, 0
        colors_rgba.append([int(red_channel), int(green_channel), int(blue_channel), 220])
    return colors_rgba


def simplify_roads_gdf_for_map_display(
    roads_gdf: gpd.GeoDataFrame,
    *,
    simplify_meters: float,
    max_features: int,
) -> gpd.GeoDataFrame:
    """Reduce vertex / feature count for GeoJsonLayer (display only; pipeline keeps full geometry)."""
    if roads_gdf.empty:
        return roads_gdf
    projected_roads_gdf = roads_gdf.to_crs(3857)
    simplified_geometry = projected_roads_gdf.geometry.simplify(simplify_meters, preserve_topology=True)
    display_roads_gdf = gpd.GeoDataFrame(geometry=simplified_geometry, crs=projected_roads_gdf.crs)
    display_roads_gdf = display_roads_gdf[~display_roads_gdf.geometry.is_empty & display_roads_gdf.geometry.notna()]
    display_roads_gdf = display_roads_gdf.to_crs(4326)
    feature_count = len(display_roads_gdf)
    cap = max(500, int(max_features))
    if feature_count > cap:
        step = max(1, math.ceil(feature_count / cap))
        display_roads_gdf = display_roads_gdf.iloc[::step].copy()
    return display_roads_gdf


def map_display_budgets(n_visible_layers: int) -> dict[str, float | int]:
    """
    When several deck.gl layers are visible, total attribute + index data grows quickly.
    Qt WebEngine (often SwiftShader) can hit GPU memory or inline-JSON practical limits; scale caps with n.
    """
    visible_layer_count = max(1, n_visible_layers)
    scale_divisor = math.sqrt(float(visible_layer_count))
    road_max = max(4000, int(ROADS_MAP_MAX_FEATURES_BASE / scale_divisor))
    road_simplify = ROADS_MAP_SIMPLIFY_METERS_BASE * (1.0 + 0.35 * (visible_layer_count - 1))
    points_max = max(8000, int(POINTS_MAP_MAX_BASE / scale_divisor))
    euler_vertices = max(40_000, int(EULER_VERTEX_BUDGET_BASE / scale_divisor))
    return {
        "road_max_features": road_max,
        "road_simplify_meters": road_simplify,
        "points_max": points_max,
        "euler_vertex_budget": euler_vertices,
    }


def euler_polylines_for_display(
    polylines: list[list[tuple[float, float]]],
    vertex_budget: int,
) -> list[list[tuple[float, float]]]:
    """Same length as input so sector index / colors stay aligned with the original route list."""
    total_vertex_count = sum(len(polyline) for polyline in polylines if len(polyline) >= 2)
    if total_vertex_count <= vertex_budget:
        return [list(polyline) for polyline in polylines]
    display_polylines: list[list[tuple[float, float]]] = []
    for polyline in polylines:
        if len(polyline) < 2:
            display_polylines.append(list(polyline))
            continue
        vertex_share = max(2, int(vertex_budget * len(polyline) / total_vertex_count))
        if len(polyline) <= vertex_share:
            display_polylines.append(list(polyline))
        else:
            sampled_indices = np.linspace(0, len(polyline) - 1, vertex_share).astype(int)
            display_polylines.append([polyline[int(vertex_index)] for vertex_index in sampled_indices])
    return display_polylines
