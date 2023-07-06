import networkx as nx
import numpy as np

from ...algorithms.static.utils import points_to_graph
from ...environment import constants as c
from ...environment.actions import Action, ALL_BASEACTIONS
from ...environment.entity.entity import Entity
from ..doors import constants as do
from ..maintenance import constants as mi
from ...utils.helpers import MOVEMAP
from ...utils.render import RenderEntity
from ...utils.states import Gamestate


class Maintainer(Entity):

    @property
    def var_can_collide(self):
        return True

    @property
    def var_can_move(self):
        return False

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_has_position(self):
        return True

    def __init__(self, state: Gamestate, objective: str, action: Action, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = action
        self.actions = [x() for x in ALL_BASEACTIONS]
        self.objective = objective
        self._path = None
        self._next = []
        self._last = []
        self._last_serviced = 'None'
        self._floortile_graph = points_to_graph(state[c.FLOOR].positions)

    def tick(self, state):
        if found_objective := state[self.objective].by_pos(self.pos):
            if found_objective.name != self._last_serviced:
                self.action.do(self, state)
                self._last_serviced = found_objective.name
            else:
                action = self.get_move_action(state)
                return action.do(self, state)
        else:
            action = self.get_move_action(state)
            return action.do(self, state)

    def get_move_action(self, state) -> Action:
        if self._path is None or not self._path:
            if not self._next:
                self._next = list(state[self.objective].values())
                self._last = []
            self._last.append(self._next.pop())
            self._path = self.calculate_route(self._last[-1])

        if door := self._door_is_close():
            if door.is_closed:
                # Translate the action_object to an integer to have the same output as any other model
                action = do.ACTION_DOOR_USE
            else:
                action = self._predict_move(state)
        else:
            action = self._predict_move(state)
        # Translate the action_object to an integer to have the same output as any other model
        try:
            action_obj = next(x for x in self.actions if x.name == action)
        except (StopIteration, UnboundLocalError):
            print('Will not happen')
            raise EnvironmentError
        return action_obj

    def calculate_route(self, entity):
        route = nx.shortest_path(self._floortile_graph, self.pos, entity.pos)
        return route[1:]

    def _door_is_close(self):
        try:
            return next(y for x in self.tile.neighboring_floor for y in x.guests if do.DOOR in y.name)
        except StopIteration:
            return None

    def _predict_move(self, state):
        next_pos = self._path[0]
        if len(state[c.FLOOR].by_pos(next_pos).guests_that_can_collide) > 0:
            action = c.NOOP
        else:
            next_pos = self._path.pop(0)
            diff = np.subtract(next_pos, self.pos)
            # Retrieve action based on the pos dif (like in: What do I have to do to get there?)
            action = next(action for action, pos_diff in MOVEMAP.items() if np.all(diff == pos_diff))
        return action

    def render(self):
        return RenderEntity(mi.MAINTAINER, self.pos)
