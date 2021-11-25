import numpy as np

from networkx.algorithms.approximation import traveling_salesman as tsp

from environments.factory.base.objects import Agent
from environments.factory.base.registers import FloorTiles, Actions
from environments.helpers import points_to_graph
from environments import helpers as h


class TSPDirtAgent(Agent):

    def __init__(self, floortiles: FloorTiles, dirt_register, actions: Actions, *args,
                 static_problem: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_problem = static_problem
        self._floortiles = floortiles
        self._actions = actions
        self._dirt_register = dirt_register
        self._floortile_graph = points_to_graph(self._floortiles.positions,
                                                allow_euclidean_connections=self._actions.allow_diagonal_movement,
                                                allow_manhattan_connections=self._actions.allow_square_movement)
        self._static_route = None

    def predict(self, *_, **__):
        if self._dirt_register.by_pos(self.pos) is not None:
            # Translate the action_object to an integer to have the same output as any other model
            action = h.EnvActions.CLEAN_UP
        elif any('door' in x.name.lower() for x in self.tile.guests):
            door = next(x for x in self.tile.guests if 'door' in x.name.lower())
            if door.is_closed:
                # Translate the action_object to an integer to have the same output as any other model
                action = h.EnvActions.USE_DOOR
            else:
                action = self._predict_move()
        else:
            action = self._predict_move()
        # Translate the action_object to an integer to have the same output as any other model
        action_obj = next(action_i for action_i, action_obj in enumerate(self._actions) if action_obj == action)
        return action_obj

    def _predict_move(self):
        if self.static_problem:
            if self._static_route is None:
                self._static_route = self.calculate_tsp_route()
            else:
                pass
            next_pos = self._static_route.pop(0)
            while next_pos == self.pos:
                next_pos = self._static_route.pop(0)
        else:
            raise NotImplementedError

        diff = np.subtract(next_pos, self.pos)
        # Retrieve action based on the pos dif (like in: What do i have to do to get there?)
        try:
            action = next(action for action, pos_diff in h.ACTIONMAP.items()
                          if (diff == pos_diff).all())
        except StopIteration:
            print('This Should not happen!')
        return action

    def calculate_tsp_route(self):
        route = tsp.traveling_salesman_problem(self._floortile_graph,
                                               nodes=[self.pos] + [x for x in self._dirt_register.positions])
        return route
