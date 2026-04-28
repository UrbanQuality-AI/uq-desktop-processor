"""
Loads and saves sampling-point GeoJSON used between pipeline stages.
"""

from pathlib import Path

import geopandas as gpd


def load_sampling_points_latlon(path: str | Path) -> list[tuple[float, float]]:
    """
    Load a point layer (Point geometries, WGS84) into ``(latitude, longitude)`` tuples
    for Mapillary download and map display.

    Non-point geometries are converted to centroids.

    :param path: Path to a vector layer containing sampling geometries.
    :return: Sampling points as ``(latitude, longitude)`` pairs.
    :raises FileNotFoundError: If the point layer file does not exist.

    Example::
        In: load_sampling_points_latlon("data/results/sampling_points.geojson")
        Out: [(50.0612, 19.9377), (50.0620, 19.9402), ...]
    """
    point_layer_path = Path(path)
    if not point_layer_path.is_file():
        raise FileNotFoundError(f"Point layer file not found: {point_layer_path}")
    gdf = gpd.read_file(point_layer_path)
    if gdf.empty:
        return []
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    latlon_points: list[tuple[float, float]] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Point":
            latlon_points.append((float(geom.y), float(geom.x)))
        else:
            # Keep a single representative coordinate for non-point input.
            centroid = geom.centroid
            latlon_points.append((float(centroid.y), float(centroid.x)))
    return latlon_points
