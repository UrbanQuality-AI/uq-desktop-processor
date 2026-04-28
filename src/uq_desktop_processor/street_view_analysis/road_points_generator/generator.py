"""
Generates evenly spaced sampling points along road network edges.
"""

import logging

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from .filtering import filter_close_points

log = logging.getLogger(__name__)


def _get_lines_from_geometry(geometry: BaseGeometry) -> list[LineString]:
    """
    Normalize geometry to a list of ``LineString`` parts.

    :param geometry: ``LineString`` or ``MultiLineString``.
    :return: Non-empty list of lines, or empty if the type is unsupported.

    Example::
        In: _get_lines_from_geometry(LineString([(0, 0), (1, 1)]))
        Out: [LineString(...)]
    """
    if isinstance(geometry, LineString):
        return [geometry]
    elif isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    return []


def _generate_points_from_lines(lines: list[LineString], spacing: float) -> list[Point]:
    """
    Place points every ``spacing`` meters along each line (including endpoints where applicable).

    :param lines: Road segments in a metric CRS.
    :param spacing: Step length in meters.
    :return: Shapely ``Point`` geometries along the lines.

    Example::
        In: _generate_points_from_lines([LineString([(0, 0), (100, 0)])], spacing=50)
        Out: [Point(...), Point(...), Point(...)]
    """
    if spacing <= 0:
        msg = "Spacing must be a positive number greater than zero."
        log.error(msg)
        raise ValueError(msg)

    points = []
    for line in lines:
        length = line.length
        num_points = int(length // spacing)
        # Include start and regular steps along the segment (metric CRS)
        points.extend([line.interpolate(step_index * spacing) for step_index in range(num_points + 1)])
    return points


def generate_points_along_roads(
    roads_gdf: gpd.GeoDataFrame, spacing: float = 100, min_distance_m: float = 20
) -> list[tuple[float, float]]:
    """
    Sample ``(latitude, longitude)`` along road line geometries, then thin by minimum distance.

    :param roads_gdf: Roads as ``LineString`` / ``MultiLineString``; reprojected to metric CRS if geographic.
    :param spacing: Along-line spacing in meters.
    :param min_distance_m: Post-process minimum separation between returned points.
    :return: List of ``(lat, lon)`` in WGS84 after :func:`filter_close_points`.

    Example::
        In: generate_points_along_roads(roads_gdf, spacing=100, min_distance_m=20)
        Out: [(50.06, 19.94), ...]
    """
    log.info(
        "Generating road points for %s geometry objects (spacing=%sm).",
        len(roads_gdf),
        spacing,
    )

    if roads_gdf.crs is None or roads_gdf.crs.is_geographic:
        log.debug("Reprojecting input GeoDataFrame to EPSG:3857.")
        roads_gdf = roads_gdf.to_crs(epsg=3857)

    generated_points = []  # Shapely Points in roads_gdf.crs (meters)

    for _, row in tqdm(roads_gdf.iterrows(), total=len(roads_gdf), desc="Generating points"):
        lines = _get_lines_from_geometry(row.geometry)
        points = _generate_points_from_lines(lines, spacing)
        generated_points.extend(points)

    log.debug(
        "Generated %s raw points. Converting to WGS84...",
        len(generated_points),
    )

    geo_series = gpd.GeoSeries(generated_points, crs=roads_gdf.crs).to_crs(epsg=4326)
    # Pipeline convention: (lat, lon)
    latlon_points = [(point.y, point.x) for point in geo_series]

    return filter_close_points(latlon_points, min_distance_m)
