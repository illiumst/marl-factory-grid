import itertools

import networkx as nx
import numpy as np


def points_to_graph(coordiniates, allow_euclidean_connections=True, allow_manhattan_connections=True):
    """
    Given a set of coordinates, this function contructs a non-directed graph, by conncting adjected points.
    There are three combinations of settings:
        Allow all neigbors:     Distance(a, b) <= sqrt(2)
        Allow only manhattan:   Distance(a, b) == 1
        Allow only Euclidean:   Distance(a, b) == sqrt(2)


    :param coordiniates: A set of coordinates.
    :type coordiniates: Tuple[int, int]
    :param allow_euclidean_connections: Whether to regard diagonal adjected cells as neighbors
    :type: bool
    :param allow_manhattan_connections: Whether to regard directly adjected cells as neighbors
    :type: bool

    :return: A graph with nodes that are conneceted as specified by the parameters.
    :rtype: nx.Graph
    """
    assert allow_euclidean_connections or allow_manhattan_connections
    possible_connections = itertools.combinations(coordiniates, 2)
    graph = nx.Graph()
    for a, b in possible_connections:
        diff = np.linalg.norm(np.asarray(a)-np.asarray(b))
        if allow_manhattan_connections and allow_euclidean_connections and diff <= np.sqrt(2):
            graph.add_edge(a, b)
        elif not allow_manhattan_connections and allow_euclidean_connections and diff == np.sqrt(2):
            graph.add_edge(a, b)
        elif allow_manhattan_connections and not allow_euclidean_connections and diff == 1:
            graph.add_edge(a, b)
    return graph
