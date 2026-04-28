"""
Load a drive network from OSM or files, filter highways, consolidate intersections, project to WGS84.
"""

import logging
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import networkx as nx
import osmnx as ox

log = logging.getLogger(__name__)

EXCLUDED_HIGHWAY_TYPES = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
    }
)


def _edge_has_excluded_highway(edge_data: dict[str, Any]) -> bool:
    """
    Return True if an edge's ``highway`` tag is in :data:`EXCLUDED_HIGHWAY_TYPES`.

    :param edge_data: OSMnx edge attribute dict.
    :return: Whether the edge should be dropped before routing.

    Example::
        In: _edge_has_excluded_highway({"highway": "motorway"})
        Out: True
    """
    highway_value = edge_data.get("highway")
    if highway_value is None:
        return False
    if isinstance(highway_value, str):
        return highway_value in EXCLUDED_HIGHWAY_TYPES
    if isinstance(highway_value, list | tuple):
        return any(
            isinstance(highway_entry, str) and highway_entry in EXCLUDED_HIGHWAY_TYPES
            for highway_entry in highway_value
        )
    return False


def load_route_aligned_graph_wgs84(
    *,
    city_name: str | None = None,
    region_geojson_path: str | Path | None = None,
    road_geojson_path: str | Path | None = None,
    consolidate_tolerance_m: float = 15.0,
    use_cache: bool = True,
) -> tuple[nx.MultiGraph, gpd.GeoDataFrame]:
    """
    Build an undirected multigraph in EPSG:4326 for routing and point sampling.

    Exactly one of ``city_name``, ``region_geojson_path``, or ``road_geojson_path`` must be set.
    Freeway/trunk edges are removed, intersections are merged, then the graph is reprojected to WGS84.

    :param city_name: OSM place query.
    :param region_geojson_path: Polygon file to download a network inside.
    :param road_geojson_path: Existing road lines to turn into a graph.
    :param consolidate_tolerance_m: Meters for ``ox.consolidate_intersections``.
    :param use_cache: OSMnx HTTP cache toggle.
    :return: ``(graph_wgs84, area_or_edges_gdf)`` for bounds or bookkeeping.
    :raises ValueError: If not exactly one source is provided.

    Example::
        In: load_route_aligned_graph_wgs84(city_name="Katowice, Poland")
        Out: (nx.MultiGraph with WGS84 node coords, GeoDataFrame describing area/edges bounds)
    """
    ox.settings.use_cache = use_cache  # type: ignore

    sources = [
        city_name is not None,
        region_geojson_path is not None,
        road_geojson_path is not None,
    ]
    if sum(sources) != 1:
        msg = "Provide exactly one of: city_name, region_geojson_path, road_geojson_path."
        raise ValueError(msg)

    # Branch 1: user-supplied line geometries -> graph
    if road_geojson_path is not None:
        road_file_path = Path(road_geojson_path)
        log.info("Loading road network from file: %s", road_file_path)
        road_edges_gdf = gpd.read_file(road_file_path)
        if road_edges_gdf.crs is None:
            road_edges_gdf = road_edges_gdf.set_crs("EPSG:4326")

        raw_graph = ox.graph_from_gdfs(gdf_nodes=cast(Any, None), gdf_edges=road_edges_gdf)
        city_gdf = road_edges_gdf.to_crs("EPSG:4326")
    # Branch 2: polygon AOI -> download inside
    elif region_geojson_path is not None:
        region_file_path = Path(region_geojson_path)
        log.info("Loading region polygon from file: %s", region_file_path)
        region_gdf = gpd.read_file(region_file_path)
        region_polygon_union = region_gdf.unary_union
        raw_graph = ox.graph_from_polygon(region_polygon_union, network_type="drive", simplify=True)
        city_gdf = region_gdf.to_crs("EPSG:4326")
    # Branch 3: named place -> download
    else:
        assert city_name is not None
        log.info("Downloading and building graph for %s", city_name)
        raw_graph = ox.graph_from_place(city_name, network_type="drive", simplify=True)
        city_gdf = ox.geocode_to_gdf(city_name)

    # Drop motorways/trunk and any nodes left isolated
    edges_to_remove = [
        (start_node, end_node, edge_key)
        for start_node, end_node, edge_key, edge_attributes in raw_graph.edges(data=True, keys=True)  # type: ignore
        if _edge_has_excluded_highway(edge_attributes)
    ]
    raw_graph.remove_edges_from(edges_to_remove)
    isolated_nodes = list(nx.isolates(raw_graph))
    raw_graph.remove_nodes_from(isolated_nodes)
    log.info("Removed %s edges and %s isolated nodes.", len(edges_to_remove), len(isolated_nodes))

    # Consolidation runs in projected (metric) CRS
    projected_graph = ox.project_graph(raw_graph)
    log.info("Consolidating intersections (tolerance=%s m)", consolidate_tolerance_m)
    consolidated_graph = ox.consolidate_intersections(
        projected_graph,
        rebuild_graph=True,
        tolerance=consolidate_tolerance_m,
        dead_ends=False,
    )

    geographic_directed_graph = ox.project_graph(consolidated_graph, to_crs="EPSG:4326")

    # Undirected multigraph for downstream algorithms
    geographic_undirected_graph = nx.MultiGraph(geographic_directed_graph)
    geographic_undirected_graph.graph.update(geographic_directed_graph.graph)  # type: ignore

    return geographic_undirected_graph, city_gdf


def route_aligned_edges_web_mercator(geographic_graph: nx.MultiGraph) -> gpd.GeoDataFrame:
    """
    Export non-duplicate road edges as a Web Mercator GeoDataFrame for metric sampling.

    :param geographic_graph: Output of :func:`load_route_aligned_graph_wgs84`.
    :return: Edge table with unique geometries in EPSG:3857.

    Example::
        In: route_aligned_edges_web_mercator(geographic_graph).crs.to_string()
        Out: "EPSG:3857"
    """
    edges = ox.graph_to_gdfs(cast(nx.MultiDiGraph, geographic_graph), nodes=False, edges=True)
    edges = edges.to_crs(epsg=3857)
    # Same physical street can appear as two directed rows
    return edges.drop_duplicates(subset="geometry")


__all__ = [
    "EXCLUDED_HIGHWAY_TYPES",
    "load_route_aligned_graph_wgs84",
    "route_aligned_edges_web_mercator",
]
