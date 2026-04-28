"""
Order edge geometries along an Eulerian circuit into a single WGS84 polyline.
"""

from typing import Any

import networkx as nx


def polyline_from_eulerian_circuit(
    eulerian_graph: nx.MultiGraph,
    circuit: list[tuple[Any, Any, Any]],
) -> list[tuple[float, float]]:
    """
    Walk the circuit and concatenate edge ``geometry`` coords (or node x/y) in order.

    Coordinates are ``(longitude, latitude)``. Duplicate consecutive vertices are skipped.

    :param eulerian_graph: Graph whose edges may carry Shapely line geometries (OSMnx-style).
    :param circuit: Eulerian circuit from ``nx.eulerian_circuit(..., keys=True)``.
    :return: Ordered vertex list along the closed walk.

    Example::
        In: polyline_from_eulerian_circuit(eulerian_graph, circuit)
        Out: [(19.94, 50.06), (19.95, 50.06), ...]
    """
    ordered_polyline: list[tuple[float, float]] = []
    for start_node, end_node, edge_key in circuit:
        edge_data = eulerian_graph.get_edge_data(start_node, end_node, edge_key)
        start_node_data = eulerian_graph.nodes[start_node]

        if edge_data is not None and "geometry" in edge_data:
            coords = list(edge_data["geometry"].coords)
            # Orient LineString to start near circuit start_node
            start_distance_squared = (coords[0][0] - start_node_data["x"]) ** 2 + (
                coords[0][1] - start_node_data["y"]
            ) ** 2
            end_distance_squared = (coords[-1][0] - start_node_data["x"]) ** 2 + (
                coords[-1][1] - start_node_data["y"]
            ) ** 2
            if end_distance_squared < start_distance_squared:
                coords = coords[::-1]
            for lon, lat in coords:
                # Append only when this vertex differs from the previous one.
                # This avoids duplicate consecutive points at segment joins.
                if not ordered_polyline or (ordered_polyline[-1][1] != lat or ordered_polyline[-1][0] != lon):
                    ordered_polyline.append((float(lon), float(lat)))
        else:
            # Straight edge: use node coordinates only
            for node_id in (start_node, end_node):
                node_data = eulerian_graph.nodes[node_id]
                node_lon = float(node_data["x"])
                node_lat = float(node_data["y"])
                # Same dedup rule for fallback node-based coordinates.
                if not ordered_polyline or (ordered_polyline[-1][1] != node_lat or ordered_polyline[-1][0] != node_lon):
                    ordered_polyline.append((node_lon, node_lat))
    return ordered_polyline
