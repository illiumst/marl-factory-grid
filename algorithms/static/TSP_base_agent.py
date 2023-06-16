import itertools
from random import choice

import numpy as np

import networkx as nx
from networkx.algorithms.approximation import traveling_salesman as tsp


from modules.doors import constants as do
from environment import constants as c
from environment.utils.helpers import MOVEMAP

from abc import abstractmethod, ABC

future_planning = 7


def points_to_graph(coordiniates_or_tiles, allow_euclidean_connections=True, allow_manhattan_connections=True):
    """
    Given a set of coordinates, this function contructs a non-directed graph, by conncting adjected points.
    There are three combinations of settings:
        Allow all neigbors:     Distance(a, b) <= sqrt(2)
        Allow only manhattan:   Distance(a, b) == 1
        Allow only euclidean:   Distance(a, b) == sqrt(2)


    :param coordiniates_or_tiles: A set of coordinates.
    :type coordiniates_or_tiles: Tiles
    :param allow_euclidean_connections: Whether to regard diagonal adjected cells as neighbors
    :type: bool
    :param allow_manhattan_connections: Whether to regard directly adjected cells as neighbors
    :type: bool

    :return: A graph with nodes that are conneceted as specified by the parameters.
    :rtype: nx.Graph
    """
    assert allow_euclidean_connections or allow_manhattan_connections
    if hasattr(coordiniates_or_tiles, 'positions'):
        coordiniates_or_tiles = coordiniates_or_tiles.positions
    possible_connections = itertools.combinations(coordiniates_or_tiles, 2)
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


class TSPBaseAgent(ABC):

    def __init__(self, state, agent_i, static_problem: bool = True):
        self.static_problem = static_problem
        self.local_optimization = True
        self._env = state
        self.state = self._env.state[c.AGENT][agent_i]
        self._floortile_graph = points_to_graph(self._env[c.FLOOR].positions)
        self._static_route = None

    @abstractmethod
    def predict(self, *_, **__) -> int:
        return 0

    def _use_door_or_move(self, door, target):
        if door.is_closed:
            # Translate the action_object to an integer to have the same output as any other model
            action = do.ACTION_DOOR_USE
        else:
            action = self._predict_move(target)
        return action

    def calculate_tsp_route(self, target_identifier):
        positions = [x for x in self._env.state[target_identifier].positions if x != c.VALUE_NO_POS]
        if self.local_optimization:
            nodes = \
                [self.state.pos] + \
                [x for x in positions if max(abs(np.subtract(x, self.state.pos))) < 3]
            try:
                while len(nodes) < 7:
                    nodes += [next(x for x in positions if x not in nodes)]
            except StopIteration:
                nodes = [self.state.pos] + positions

        else:
            nodes = [self.state.pos] + positions
        route = tsp.traveling_salesman_problem(self._floortile_graph,
                                               nodes=nodes, cycle=True, method=tsp.greedy_tsp)
        return route

    def _door_is_close(self):
        try:
            return next(y for x in self.state.tile.neighboring_floor for y in x.guests if do.DOOR in y.name)
        except StopIteration:
            return None

    def _has_targets(self, target_identifier):
        return bool(len([x for x in self._env.state[target_identifier] if x.pos != c.VALUE_NO_POS]) >= 1)

    def _predict_move(self, target_identifier):
        if self._has_targets(target_identifier):
            if self.static_problem:
                if not self._static_route:
                    self._static_route = self.calculate_tsp_route(target_identifier)
                else:
                    pass
                next_pos = self._static_route.pop(0)
                while next_pos == self.state.pos:
                    next_pos = self._static_route.pop(0)
            else:
                if not self._static_route:
                    self._static_route = self.calculate_tsp_route(target_identifier)[:7]
                next_pos = self._static_route.pop(0)
                while next_pos == self.state.pos:
                    next_pos = self._static_route.pop(0)

            diff = np.subtract(next_pos, self.state.pos)
            # Retrieve action based on the pos dif (like in: What do I have to do to get there?)
            try:
                action = next(action for action, pos_diff in MOVEMAP.items() if np.all(diff == pos_diff))
            except StopIteration:
                print(f'diff: {diff}')
                print('This Should not happen!')
                action = choice(self.state.actions).name
        else:
            action = choice(self.state.actions).name
        # noinspection PyUnboundLocalVariable
        return action
