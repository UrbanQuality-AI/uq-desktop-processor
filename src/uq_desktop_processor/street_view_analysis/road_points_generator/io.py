"""
Reads/writes sampling-point layers and loads route-aligned road networks.
"""

import logging

import geopandas as gpd
import requests
from fiona.errors import DriverError, FionaValueError
from pyogrio.errors import DataSourceError
from yaspin import yaspin

from uq_desktop_processor.street_view_analysis.road_graph_prepare import (
    load_route_aligned_graph_wgs84,
    route_aligned_edges_web_mercator,
)

log = logging.getLogger(__name__)


def load_roads_route_aligned(
    *,
    place_name: str | None = None,
    region_geojson_path: str | None = None,
    road_geojson_path: str | None = None,
    consolidate_tolerance_m: float = 15.0,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Road edges in EPSG:3857 from the same graph pipeline as Euler routes (drive, no trunk/motorway).

    :param place_name: OSM place query, if used alone.
    :param region_geojson_path: Region polygon file, if used alone.
    :param road_geojson_path: Local road lines file, if used alone.
    :param consolidate_tolerance_m: ``ox.consolidate_intersections`` tolerance in meters.
    :param use_cache: Whether OSMnx may use its HTTP cache.
    :return: Edge GeoDataFrame in Web Mercator.

    Example::
        In: load_roads_route_aligned(place_name="Katowice, Poland").crs.to_string()
        Out: "EPSG:3857"
    """
    # Shared graph build with Euler / sampling; then edge geometries only
    geographic_graph, _ = load_route_aligned_graph_wgs84(
        city_name=place_name,
        region_geojson_path=region_geojson_path,
        road_geojson_path=road_geojson_path,
        consolidate_tolerance_m=consolidate_tolerance_m,
        use_cache=use_cache,
    )
    return route_aligned_edges_web_mercator(geographic_graph)


def download_road_network(
    place_name: str,
    *,
    consolidate_tolerance_m: float = 15.0,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load route-aligned roads for a named place (same settings as GPX route generation).

    :param place_name: OSM geocodable place string.
    :param consolidate_tolerance_m: Intersection merge tolerance in meters.
    :param use_cache: OSMnx cache flag.
    :return: Metric road edges.
    :raises ValueError: If no roads are found for the place.
    :raises ConnectionError: On Overpass/network failure.
    :raises RuntimeError: On other load failures.

    Example::
        In: download_road_network("Katowice, Poland")
        Out: GeoDataFrame with LineString/MultiLineString road edges in EPSG:3857
    """
    log.info("Initiating route-aligned road network download for: '%s'", place_name)

    with yaspin(text=f"Downloading road network for: {place_name}..."):
        try:
            edges = load_roads_route_aligned(
                place_name=place_name,
                consolidate_tolerance_m=consolidate_tolerance_m,
                use_cache=use_cache,
            )
            log.info(
                "Successfully loaded %s road segments for '%s' (route-aligned).",
                len(edges),
                place_name,
            )
            return edges

        except ValueError as error:
            log.error("No road data or invalid location: %s: %s", place_name, error)
            raise ValueError(f"No roads found for: {place_name}") from error

        except requests.exceptions.RequestException as error:
            log.error("Network error when downloading data for '%s': %s", place_name, error)
            raise ConnectionError("Connection problem with Overpass API") from error

        except Exception as error:
            log.error("Unexpected error when downloading road network for '%s': %s", place_name, error)
            raise RuntimeError(f"Error while downloading road network for: {place_name}") from error


def read_road_geojson(input_path: str) -> gpd.GeoDataFrame:
    """
    Load line geometries from a local GeoJSON (no route-aligned graph build).

    For Euler-aligned sampling, prefer :func:`load_roads_route_aligned` with ``road_geojson_path``.

    :param input_path: Path to a vector file readable by GeoPandas.
    :return: Subset of features that are ``LineString`` or ``MultiLineString``.
    :raises ValueError: If the file is invalid.
    :raises FileNotFoundError: If the path does not exist.
    :raises RuntimeError: On unexpected read errors.

    Example::
        In: read_road_geojson("roads.geojson").geometry.type.isin(["LineString", "MultiLineString"]).all()
        Out: True
    """
    log.info("Reading local road network from: %s", input_path)

    try:
        roads_geodataframe = gpd.read_file(input_path)
        # Ignore points/polygons if the file is mixed
        line_geometries = roads_geodataframe[roads_geodataframe.geometry.type.isin(["LineString", "MultiLineString"])]

        log.debug("Loaded %s valid road geometries from file.", len(line_geometries))
        return line_geometries

    except (FionaValueError, DriverError, OSError) as error:
        log.error("Error reading GeoJSON: %s", error)
        raise ValueError(f"Invalid GeoJSON file: {input_path}") from error

    except DataSourceError as error:
        if "No such file or directory" in str(error):
            log.error("File does not exist: %s", input_path)
            raise FileNotFoundError(f"File not found: {input_path}") from error
        log.error("Error reading GeoJSON: %s", error)
        raise ValueError(f"Invalid GeoJSON file: {input_path}") from error

    except Exception as error:
        log.error("Unexpected error while loading file: %s", error)
        raise RuntimeError(f"Error while loading GeoJSON file: {input_path}") from error


def download_roads_from_region(
    region_path: str,
    *,
    consolidate_tolerance_m: float = 15.0,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load route-aligned roads clipped to the first polygon in a region file.

    :param region_path: GeoJSON (or similar) with a polygon geometry.
    :param consolidate_tolerance_m: Intersection merge tolerance in meters.
    :param use_cache: OSMnx cache flag.
    :return: Metric road edges for the region.
    :raises RuntimeError: If the region file cannot be processed.

    Example::
        In: download_roads_from_region("region.geojson")
        Out: GeoDataFrame with route-aligned road edges for the region (EPSG:3857)
    """
    log.info("Downloading route-aligned road network for region: %s", region_path)

    try:
        region_geodataframe = gpd.read_file(region_path)

        if region_geodataframe.empty or region_geodataframe.geometry.iloc[0] is None:
            raise ValueError(f"The provided region file is empty or invalid: {region_path}")

        # First feature defines the AOI (WGS84 for OSMnx)
        polygon = region_geodataframe.to_crs(epsg=4326).geometry.iloc[0]

        if polygon.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Expected a Polygon geometry in {region_path}, but found {polygon.geom_type}.")

        with yaspin(text=f"Downloading road network for region: {region_path}..."):
            road_edges = load_roads_route_aligned(
                region_geojson_path=region_path,
                consolidate_tolerance_m=consolidate_tolerance_m,
                use_cache=use_cache,
            )
            log.info("Successfully loaded %s road segments for custom region.", len(road_edges))
            return road_edges

    except ValueError:
        raise
    except Exception as error:
        log.error("Error processing region file '%s': %s", region_path, error)
        raise RuntimeError(f"Failed to download roads for region: {region_path}") from error
