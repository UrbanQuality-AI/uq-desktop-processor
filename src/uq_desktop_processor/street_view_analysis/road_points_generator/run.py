"""
CLI-oriented pipeline: load route-aligned roads, sample points, write a layer file.
"""

import logging
from pathlib import Path

import geopandas as gpd

from .generator import generate_points_along_roads
from .io import load_roads_route_aligned

log = logging.getLogger(__name__)


def _save_sampling_points_layer(
    points: list[tuple[float, float]],
    output_path: str | Path,
) -> Path:
    """
    Write sampling points to a vector file in EPSG:4326.

    Coordinates follow :func:`generate_points_along_roads`: each tuple is ``(latitude, longitude)``.

    :param points: Sample locations.
    :param output_path: GeoPackage, GeoJSON, or other OGR path.
    :return: Resolved path that was written.

    Example::
        In: _save_sampling_points_layer([(50.06, 19.94)], "data/results/points.geojson")
        Out: Path(".../points.geojson")
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        points_gdf = gpd.GeoDataFrame({"id": []}, geometry=gpd.GeoSeries([], crs="EPSG:4326"))
    else:
        latitudes, longitudes = zip(*points, strict=False)
        # points_from_xy: x=lon, y=lat (EPSG:4326)
        points_gdf = gpd.GeoDataFrame(
            {"id": range(len(points))},
            geometry=gpd.points_from_xy(longitudes, latitudes),
            crs="EPSG:4326",
        )
    points_gdf.to_file(output_path_obj)
    return output_path_obj.resolve()


def build_points_pipeline(
    *,
    output_points_path: str | Path,
    place_name: str | None = None,
    region_geojson_path: str | None = None,
    road_geojson_path: str | None = None,
    spacing: float = 100,
    min_distance_m: float = 20,
    consolidate_tolerance_m: float = 15.0,
    use_cache: bool = True,
) -> tuple[gpd.GeoDataFrame, list[tuple[float, float]]]:
    """
    End-to-end sampling: exactly one of place, region file, or road file; then generate and save points.

    :param output_points_path: Output vector path for point features.
    :param place_name: OSM place name (mutually exclusive with region/road paths).
    :param region_geojson_path: Polygon file for a custom AOI.
    :param road_geojson_path: Predefined road lines file.
    :param spacing: Along-road spacing in meters.
    :param min_distance_m: Minimum spacing after global thinning.
    :param consolidate_tolerance_m: Same as Euler route graph preparation.
    :param use_cache: OSMnx HTTP cache.
    :return: ``(roads_gdf, list of (lat, lon))``.
    :raises ValueError: If zero or more than one source argument is given.

    Example::
        In: build_points_pipeline(place_name="Katowice, Poland", output_points_path="out/points.geojson")
        Out: (road_edges_gdf, [(50.06, 19.94), (50.0612, 19.9415), ...])
    """
    log.info(
        "Starting points generation pipeline (spacing=%sm, min_dist=%sm, consolidate=%sm).",
        spacing,
        min_distance_m,
        consolidate_tolerance_m,
    )

    # Exactly one OSM / file source for load_route_aligned_graph_wgs84
    sources_provided = sum(
        source_value is not None for source_value in [place_name, region_geojson_path, road_geojson_path]
    )

    if sources_provided != 1:
        msg = (
            "You must provide exactly one data source: " "`place_name`, `region_geojson_path`, OR `road_geojson_path`."
        )
        log.error(msg)
        raise ValueError(msg)

    if place_name is not None:
        log.info("Source selected: place name '%s'", place_name)
    elif region_geojson_path is not None:
        log.info("Source selected: region polygon '%s'", region_geojson_path)
    else:
        log.info("Source selected: road GeoJSON '%s'", road_geojson_path)

    roads = load_roads_route_aligned(
        place_name=place_name,
        region_geojson_path=region_geojson_path,
        road_geojson_path=road_geojson_path,
        consolidate_tolerance_m=consolidate_tolerance_m,
        use_cache=use_cache,
    )

    points = generate_points_along_roads(
        roads,
        spacing=spacing,
        min_distance_m=min_distance_m,
    )

    saved_output_path = _save_sampling_points_layer(points, output_points_path)
    log.info("Pipeline finished. Total points generated: %s. Saved to %s", len(points), saved_output_path)

    return roads, points
