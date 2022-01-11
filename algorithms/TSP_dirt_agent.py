import numpy as np

from networkx.algorithms.approximation import traveling_salesman as tsp

from environments.factory.base.objects import Agent
from environments.helpers import points_to_graph
from environments import helpers as h

from environments.helpers import Constants as BaseConstants
from environments.helpers import EnvActions as BaseActions


class Constants(BaseConstants):
    DIRT = 'Dirt'


class Actions(BaseActions):
    CLEAN_UP = 'do_cleanup_action'


a = Actions
c = Constants

future_planning = 7


class TSPDirtAgent(Agent):

    def __init__(self, env, *args,
                 static_problem: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_problem = static_problem
        self.local_optimization = True
        self._env = env
        self._floortile_graph = points_to_graph(self._env[c.FLOOR].positions,
                                                allow_euclidean_connections=self._env._actions.allow_diagonal_movement,
                                                allow_manhattan_connections=self._env._actions.allow_square_movement)
        self._static_route = None

    def predict(self, *_, **__):
        if self._env[c.DIRT].by_pos(self.pos) is not None:
            # Translate the action_object to an integer to have the same output as any other model
            action = a.CLEAN_UP
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
        action_obj = next(action_i for action_name, action_i in self._env.named_action_space.items() if action_name == action)
        return action_obj

    def _predict_move(self):
        if len(self._env[c.DIRT]) >= 1:
            if self.static_problem:
                if not self._static_route:
                    self._static_route = self.calculate_tsp_route()
                else:
                    pass
                next_pos = self._static_route.pop(0)
                while next_pos == self.pos:
                    next_pos = self._static_route.pop(0)
            else:
                if not self._static_route:
                    self._static_route = self.calculate_tsp_route()[:7]
                next_pos = self._static_route.pop(0)
                while next_pos == self.pos:
                    next_pos = self._static_route.pop(0)

            diff = np.subtract(next_pos, self.pos)
            # Retrieve action based on the pos dif (like in: What do i have to do to get there?)
            try:
                action = next(action for action, pos_diff in h.ACTIONMAP.items()
                              if (diff == pos_diff).all())
            except StopIteration:
                print('This Should not happen!')
        else:
            action = int(np.random.randint(self._env.action_space.n))
        return action

    def calculate_tsp_route(self):
        if self.local_optimization:
            nodes = \
                [self.pos] + \
                [x for x in self._env[c.DIRT].positions if max(abs(np.subtract(x, self.pos))) < 3]
            try:
                while len(nodes) < 7:
                    nodes += [next(x for x in self._env[c.DIRT].positions if x not in nodes)]
            except StopIteration:
                nodes = [self.pos] + self._env[c.DIRT].positions

        else:
            nodes = [self.pos] + self._env[c.DIRT].positions
        route = tsp.traveling_salesman_problem(self._floortile_graph,
                                               nodes=nodes, cycle=True, method=tsp.greedy_tsp)
        return route
