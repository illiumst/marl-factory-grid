from random import shuffle

import networkx as nx
import numpy as np

from ...environment import constants as c
from ...environment.actions import Action, ALL_BASEACTIONS
from ...environment.entity.entity import Entity
from ..doors import constants as do
from ..maintenance import constants as mi
from ...utils import helpers as h
from ...utils.utility_classes import RenderEntity, Floor
from ..doors import DoorUse


class Maintainer(Entity):

    def __init__(self, objective, action, *args, **kwargs):
        """
        Represents the maintainer entity that aims to maintain machines.
        
        :param objective: The maintainer's objective, e.g., "Machines".
        :type objective: str
        :param action: The default action to be performed by the maintainer.
        :type action: Action
        """
        super().__init__(*args, **kwargs)
        self.action = action
        self.actions = [x() for x in ALL_BASEACTIONS] + [DoorUse()]
        self.objective = objective
        self._path = None
        self._next = []
        self._last = []
        self._last_serviced = 'None'

    def tick(self, state):
        """
        If there is an objective at the current position, the maintainer performs its action on the objective.
        If the objective has changed since the last servicing, the maintainer performs the action and updates
        the last serviced objective. Otherwise, it calculates a move action and performs it.

        :param state: The current game state.
        :type state: GameState
        :return: The result of the action performed by the maintainer.
        :rtype: ActionResult
        """
        if found_objective := h.get_first(state[self.objective].by_pos(self.pos)):
            if found_objective.name != self._last_serviced:
                result = self.action.do(self, state)
                self._last_serviced = found_objective.name
            else:
                action = self.get_move_action(state)
                result = action.do(self, state)
        else:
            action = self.get_move_action(state)
            result = action.do(self, state)
        self.set_state(result)
        return result

    def set_state(self, action_result):
        """
        Updates the maintainers own status with an action result.
        """
        self._status = action_result

    def get_move_action(self, state) -> Action:
        """
        Retrieves the next move action for the agent.

        If a path is not already determined, the agent calculates the shortest path to its objective, considering doors
        and obstacles. If a closed door is found in the calculated path, the agent attempts to open it.

        :param state: The current state of the environment.
        :type state: GameState

        :return: The chosen move action for the agent.
        :rtype: Action
        """
        if self._path is None or not len(self._path):
            if not self._next:
                self._next = list(state[self.objective].values()) + [Floor(*state.random_free_position)]
                shuffle(self._next)
                self._last = []
            self._last.append(self._next.pop())
            state.print("Calculating shortest path....")
            self._path = self.calculate_route(self._last[-1], state.floortile_graph)
            if not self._path:
                self._last.append(self._next.pop())
                state.print("Calculating shortest path.... Again....")
                self._path = self.calculate_route(self._last[-1], state.floortile_graph)

        if door := self._closed_door_in_path(state):
            state.print(f"{self} found {door} that is closed. Attempt to open.")
            # Translate the action_object to an integer to have the same output as any other model
            action = do.ACTION_DOOR_USE
        else:
            action = self._predict_move(state)
        # Translate the action_object to an integer to have the same output as any other model
        try:
            action_obj = h.get_first(self.actions, lambda x: x.name == action)
        except (StopIteration, UnboundLocalError):
            print('Will not happen')
            raise EnvironmentError
        return action_obj

    def calculate_route(self, entity, floortile_graph) -> list:
        """
        :returns: path, include both the source and target position
        :rtype: list
        """
        route = nx.shortest_path(floortile_graph, self.pos, entity.pos)
        return route[1:]

    def _closed_door_in_path(self, state):
        """
        Internal Use
        """
        if self._path:
            return h.get_first(state[do.DOORS].by_pos(self._path[0]), lambda x: x.is_closed)
        else:
            return None

    def _predict_move(self, state) -> Action:
        """
        Internal Use
        """
        next_pos = self._path[0]
        if any(x for x in state.entities.pos_dict[next_pos] if x.var_can_collide) > 0:
            action = c.NOOP
        else:
            next_pos = self._path.pop(0)
            diff = np.subtract(next_pos, self.pos)
            # Retrieve action based on the pos dif (like in: What do I have to do to get there?)
            action = next(action for action, pos_diff in h.MOVEMAP.items() if np.all(diff == pos_diff))
        return action

    def render(self):
        return RenderEntity(mi.MAINTAINER, self.pos)
