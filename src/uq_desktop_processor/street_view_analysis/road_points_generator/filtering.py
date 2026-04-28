"""
Filters and deduplicates candidate road sampling coordinates.
"""

import logging

import geopandas as gpd
import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)


def _compute_projected_coords(
    points: list[tuple[float, float]], crs_from: str = "EPSG:4326", crs_to: str = "EPSG:3857"
) -> npt.NDArray[np.float64]:
    """
    Project geographic ``(lat, lon)`` points to metric coordinates for distance checks.

    :param points: Non-empty list of ``(latitude, longitude)`` tuples.
    :param crs_from: Source CRS (default WGS84).
    :param crs_to: Target CRS in meters (default Web Mercator).
    :return: Array of shape ``(n, 2)`` with ``(x, y)`` in meters.
    :raises ValueError: If ``points`` is empty or invalid.
    :raises RuntimeError: If GeoPandas projection fails.

    Example::
        In: _compute_projected_coords([(50.06, 19.94)]).shape
        Out: (1, 2)
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "Projecting %s points from %s to %s.",
            len(points),
            crs_from,
            crs_to,
        )

    try:
        latitudes, longitudes = zip(*points, strict=False)
    except ValueError as error:
        log.error("Input `points` list is invalid or empty.")
        raise ValueError("Expected `points` to be a non-empty list of (lat, lon) tuples.") from error

    try:
        # points_from_xy expects (x=lon, y=lat) in geographic CRS
        projected_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(longitudes, latitudes), crs=crs_from).to_crs(
            crs_to
        )
    except Exception as projection_error:
        log.error("Coordinate projection failed: %s", projection_error)
        raise RuntimeError("Coordinate projection failed.") from projection_error

    x_coords = projected_gdf.geometry.x
    y_coords = projected_gdf.geometry.y
    return np.column_stack((x_coords, y_coords))


def _find_sparse_indices(points: npt.NDArray[np.float64], min_distance_m: float) -> list[int]:
    """
    Greedy index selection so kept points are at least ``min_distance_m`` apart (planar).

    Uses a grid hash to limit neighbor checks.

    :param points: Projected ``(x, y)`` coordinates in meters.
    :param min_distance_m: Minimum separation between any two kept points.
    :return: Indices into ``points`` of the kept samples (in traversal order).

    Example::
        In: _find_sparse_indices(np.array([[0.0, 0.0], [1.0, 1.0], [50.0, 50.0]]), 20)
        Out: [0, 2]
    """
    # Cell side ~ min_distance_m -> only compare within a 3x3 neighborhood
    grid_coords = np.floor(points / min_distance_m).astype(int)

    occupied_cells: dict[tuple[int, int], int] = {}
    kept_indices: list[int] = []

    # Self + 8 neighbors in grid space
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    for point_index, (grid_x, grid_y) in enumerate(grid_coords):
        too_close = False
        for delta_x, delta_y in neighbor_offsets:
            neighbor_cell = (grid_x + delta_x, grid_y + delta_y)
            if neighbor_cell in occupied_cells:
                neighbor_idx = occupied_cells[neighbor_cell]
                distance = np.linalg.norm(points[point_index] - points[neighbor_idx])
                if distance < min_distance_m:
                    too_close = True
                    break

        if not too_close:
            kept_indices.append(point_index)
            occupied_cells[(grid_x, grid_y)] = point_index

    return kept_indices


def filter_close_points(points: list[tuple[float, float]], min_distance_m: float = 20) -> list[tuple[float, float]]:
    """
    Drop points that lie closer than ``min_distance_m`` to an already kept point.

    :param points: ``(latitude, longitude)`` candidates.
    :param min_distance_m: Minimum spacing in meters (Web Mercator plane).
    :return: Subset of ``points`` in original order, thinned by the rule above.

    Example::
        In: filter_close_points([(50.06, 19.94), (50.06001, 19.94001)], min_distance_m=20)
        Out: [(50.06, 19.94)]
    """
    if not points:
        log.debug("No points provided for filtering.")
        return []

    if min_distance_m <= 0:
        msg = "`min_distance_m` must be a positive number."
        log.error(msg)
        raise ValueError(msg)

    # Planar distances in meters (Web Mercator)
    coords = _compute_projected_coords(points)
    kept_indices = _find_sparse_indices(coords, min_distance_m)
    filtered_points = [points[kept_point_index] for kept_point_index in kept_indices]

    log.info(
        "Point filtering complete: %s -> %s points kept (min_dist=%sm).",
        len(points),
        len(filtered_points),
        min_distance_m,
    )

    return filtered_points
