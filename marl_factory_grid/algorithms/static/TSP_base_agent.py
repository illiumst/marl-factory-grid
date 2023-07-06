from random import choice

import numpy as np

from networkx.algorithms.approximation import traveling_salesman as tsp

from marl_factory_grid.algorithms.static.utils import points_to_graph
from marl_factory_grid.modules.doors import constants as do
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.helpers import MOVEMAP

from abc import abstractmethod, ABC

future_planning = 7


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
