"""
Grid-sector Euler routes: truncate graph per tile, Chinese postman per component, write GPX.
"""

import logging
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import osmnx as ox
from osmnx.truncate import truncate_graph_bbox

from uq_desktop_processor.street_view_analysis.road_graph_prepare import load_route_aligned_graph_wgs84

from .gpx_export import build_gpx, write_gpx
from .postman import chinese_postman_polyline
from .result import EulerRoutesResult

log = logging.getLogger(__name__)


def generate_clean_routes(
    city_name: str = "Katowice, Poland",
    grid_size: tuple[int, int] = (3, 3),
    output_dir: str | Path = "routes_chinese_postman_gpx",
    *,
    consolidate_tolerance_m: float = 15.0,
    use_cache: bool = True,
    region_geojson_path: str | Path | None = None,
    road_geojson_path: str | Path | None = None,
) -> EulerRoutesResult:
    """
    Split the study area into a lat/lon grid, run Chinese postman on each sector, emit GPX files.

    :param city_name: Default place when no region or road file is passed.
    :param grid_size: ``(columns, rows)`` of bounding-box sectors.
    :param output_dir: Directory for ``trasa_*.gpx`` outputs.
    :param consolidate_tolerance_m: Same as :func:`load_route_aligned_graph_wgs84`.
    :param use_cache: OSMnx HTTP cache.
    :param region_geojson_path: Optional polygon AOI instead of ``city_name``.
    :param road_geojson_path: Optional local roads instead of downloading.
    :return: Resolved paths and polylines for all successful sectors.

    Example::
        In: generate_clean_routes(city_name="Katowice, Poland", grid_size=(2, 2)).gpx_paths
        Out: tuple of saved GPX paths, e.g. (Path(".../trasa_1.gpx"), Path(".../trasa_2.gpx"), ...)
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    geographic_graph, city_gdf = load_route_aligned_graph_wgs84(
        city_name=city_name,
        region_geojson_path=region_geojson_path,
        road_geojson_path=road_geojson_path,
        consolidate_tolerance_m=consolidate_tolerance_m,
        use_cache=use_cache,
    )

    # OSMnx truncate_* expects a MultiDiGraph (successors/predecessors); load_route_aligned_graph_wgs84 returns MultiGraph.
    directed_geographic_graph = nx.MultiDiGraph(geographic_graph)
    directed_geographic_graph.graph.update(geographic_graph.graph)

    w_bound, s_bound, e_bound, n_bound = city_gdf.total_bounds
    column_count, row_count = grid_size[0], grid_size[1]
    # Grid lines in WGS84; each cell becomes one sector
    lon_grid_boundaries = np.linspace(w_bound, e_bound, column_count + 1)
    lat_grid_boundaries = np.linspace(s_bound, n_bound, row_count + 1)

    sectors: list[dict[str, float]] = []
    for lat_index in range(len(lat_grid_boundaries) - 1):
        for lon_index in range(len(lon_grid_boundaries) - 1):
            sectors.append(
                {
                    "s": float(lat_grid_boundaries[lat_index]),
                    "n": float(lat_grid_boundaries[lat_index + 1]),
                    "w": float(lon_grid_boundaries[lon_index]),
                    "e": float(lon_grid_boundaries[lon_index + 1]),
                }
            )

    gpx_paths: list[Path] = []
    polylines: list[tuple[tuple[float, float], ...]] = []

    for sector_index, sector_bounds in enumerate(sectors):
        tile_number = sector_index + 1
        log.info("Sector %s / %s", tile_number, len(sectors))

        try:
            # OSMnx bbox order: north, south, east, west
            bounding_box = (sector_bounds["n"], sector_bounds["s"], sector_bounds["e"], sector_bounds["w"])
            sector_graph = truncate_graph_bbox(directed_geographic_graph, bbox=bounding_box, truncate_by_edge=True)

            if sector_graph is None or len(sector_graph.nodes) < 2:
                log.warning("Sector %s: empty graph, skipping.", tile_number)
                continue

            undirected_sector_graph = ox.get_undirected(cast(nx.MultiDiGraph, sector_graph))
            connected_components = list(nx.connected_components(undirected_sector_graph))
            connected_components.sort(key=len, reverse=True)  # Larger components first (typical main roads)

            sector_polylines: list[list[tuple[float, float]]] = []

            for component_index, component_nodes in enumerate(connected_components, start=1):
                component_subgraph = undirected_sector_graph.subgraph(component_nodes).copy()
                if component_subgraph.number_of_edges() < 1:
                    continue
                try:
                    route_polyline = chinese_postman_polyline(cast(nx.MultiGraph, component_subgraph))
                except Exception as component_error:
                    log.exception(
                        "Sector %s subgraph %s / %s: Chinese postman failed: %s",
                        tile_number,
                        component_index,
                        len(connected_components),
                        component_error,
                    )
                    continue
                if len(route_polyline) < 2:
                    continue
                sector_polylines.append(route_polyline)

            if not sector_polylines:
                log.warning("Sector %s: no routable subgraphs, skipping file.", tile_number)
                continue

            # Inform when one tile produced multiple disjoint walks
            # Count only components that actually contain edges (ignore isolated-node components).
            component_count_with_edges = sum(
                1
                for component_nodes in connected_components
                if undirected_sector_graph.subgraph(component_nodes).number_of_edges() >= 1
            )
            if component_count_with_edges > 1:
                log.info(
                    "Sector %s: %s route(s) covering %s disconnected subgraph(s) with edges.",
                    tile_number,
                    len(sector_polylines),
                    component_count_with_edges,
                )

            gpx_document = build_gpx(sector_polylines, sector_name=f"Sector {tile_number}")
            gpx_file_path = output_directory / f"route_{tile_number}.gpx"
            write_gpx(gpx_file_path, gpx_document)
            log.info("Wrote %s", gpx_file_path)

            gpx_paths.append(gpx_file_path)
            polylines.extend(tuple(polyline) for polyline in sector_polylines)

        except ValueError as error:
            # Empty bbox / polygon (e.g. sector outside drivable network after simplify)
            if "no graph nodes" in str(error).lower():
                log.warning("Sector %s: no graph nodes in sector bounds, skipping.", tile_number)
                continue
            raise
        except Exception as error:
            log.exception("Error in sector %s: %s", tile_number, error)

    return EulerRoutesResult(
        output_dir=output_directory.resolve(),
        gpx_paths=tuple(gpx_paths),
        polylines_wgs84=tuple(polylines),
    )


__all__ = ["generate_clean_routes"]
