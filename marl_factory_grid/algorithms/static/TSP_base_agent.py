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
        """
        Abstract base class for agents in the environment.

        :param state: The environment state
        :type state:
        :param agent_i: Index of the agent
        :type agent_i: int
        :param static_problem: Indicates whether the TSP is a static problem. (Default: True)
        :type static_problem: bool
        """
        self.static_problem = static_problem
        self.local_optimization = True
        self._env = state
        self.state = self._env.state[c.AGENT][agent_i]
        self._position_graph = points_to_graph(self._env.entities.floorlist)
        self._static_route = None

    @abstractmethod
    def predict(self, *_, **__) -> int:
        """
        Predicts the next action based on the environment state.

        :return: Predicted action.
        :rtype: int
        """
        return 0

    def _use_door_or_move(self, door, target):
        """
        Helper method to decide whether to use a door or move towards a target.

        :param door: Door entity.
        :type door: Door
        :param target: Target type. For example 'Dirt', 'Dropoff' or 'Destination'
        :type target: str

        :return: Action to perform (use door or move).
        """
        if door.is_closed:
            # Translate the action_object to an integer to have the same output as any other model
            action = do.ACTION_DOOR_USE
        else:
            action = self._predict_move(target)
        return action

    def calculate_tsp_route(self, target_identifier):
        """
        Calculate the TSP route to reach a target.

        :param target_identifier: Identifier of the target entity
        :type target_identifier: str

        :return: TSP route
        :rtype: List[int]
        """
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
        route = tsp.traveling_salesman_problem(self._position_graph,
                                               nodes=nodes, cycle=True, method=tsp.greedy_tsp)
        return route

    def _door_is_close(self, state):
        """
        Check if a door is close to the agent's position.

        :param state: Current environment state.
        :type state: Gamestate

        :return: Closest door entity or None if no door is close.
        :rtype: Door | None
        """
        try:
            return next(y for x in state.entities.neighboring_positions(self.state.pos)
                        for y in state.entities.pos_dict[x] if do.DOOR in y.name)
        except StopIteration:
            return None

    def _has_targets(self, target_identifier):
        """
        Check if there are targets available in the environment.

        :param target_identifier: Identifier of the target entity.
        :type target_identifier: str

        :return: True if there are targets, False otherwise.
        :rtype: bool
        """
        return bool(len([x for x in self._env.state[target_identifier] if x.pos != c.VALUE_NO_POS]) >= 1)

    def _predict_move(self, target_identifier):
        """
           Predict the next move based on the given target.

           :param target_identifier: Identifier of the target entity.
           :type target_identifier: str

           :return: Predicted action.
           :rtype: int
           """
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
