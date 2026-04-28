"""
Chinese postman: duplicate edges to balance odd degrees, then extract an Euler tour polyline.
"""

from typing import Any

import networkx as nx

from .polyline import polyline_from_eulerian_circuit


def chinese_postman_polyline(undirected_graph: nx.MultiGraph) -> list[tuple[float, float]]:
    """
    Minimum-length closed walk covering every edge at least once (undirected, connected, with edges).

    Odd-degree nodes are paired by min-weight matching on shortest-path distances; duplicated
    edges are added, then an Euler circuit is converted to a ``(lon, lat)`` polyline.

    :param undirected_graph: Road subgraph; edge attribute ``length`` weights paths and matching.
    :return: Closed polyline in WGS84 node order.

    Example::
        In: chinese_postman_polyline(undirected_graph)
        Out: [(19.94, 50.06), (19.95, 50.06), ...]
    """
    odd_degree_nodes = [node_id for node_id, degree in undirected_graph.degree() if degree % 2 == 1]
    odd_node_matching: Any = []
    if odd_degree_nodes:
        # Pair odd vertices by shortest-path length; min matching yields min duplicate mileage
        odd_complete_graph = nx.Graph()
        for odd_node_index, source_node in enumerate(odd_degree_nodes):
            distance_by_node = nx.single_source_dijkstra_path_length(undirected_graph, source_node, weight="length")
            for target_node in odd_degree_nodes[odd_node_index + 1 :]:
                if target_node in distance_by_node:
                    odd_complete_graph.add_edge(source_node, target_node, weight=distance_by_node[target_node])
        odd_node_matching = nx.algorithms.matching.min_weight_matching(odd_complete_graph, weight="weight")

    # Start from original edges; duplicate along shortest paths to make all degrees even
    eulerian_graph = nx.MultiGraph(undirected_graph)
    for source_node, target_node in odd_node_matching:
        shortest_path_nodes = nx.shortest_path(undirected_graph, source_node, target_node, weight="length")
        for path_index in range(len(shortest_path_nodes) - 1):
            path_start_node, path_end_node = shortest_path_nodes[path_index], shortest_path_nodes[path_index + 1]
            edge_data = undirected_graph.get_edge_data(path_start_node, path_end_node)
            # MultiGraph: pick first parallel edge's attrs when duplicating
            edge_attributes = edge_data[0] if isinstance(edge_data, dict) and 0 in edge_data else edge_data
            if isinstance(edge_attributes, dict):
                eulerian_graph.add_edge(path_start_node, path_end_node, **edge_attributes)
            else:
                eulerian_graph.add_edge(path_start_node, path_end_node)

    eulerian_circuit_edges = list(nx.eulerian_circuit(eulerian_graph, keys=True))
    return polyline_from_eulerian_circuit(eulerian_graph, eulerian_circuit_edges)
