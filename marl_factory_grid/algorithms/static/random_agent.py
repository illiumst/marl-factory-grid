from random import randint

from marl_factory_grid.algorithms.static.TSP_base_agent import TSPBaseAgent

future_planning = 7


class TSPRandomAgent(TSPBaseAgent):

    def __init__(self, n_actions, *args, **kwargs):
        super(TSPRandomAgent, self).__init__(*args, **kwargs)
        self.n_action = n_actions

    def predict(self, *_, **__):
        return randint(0, self.n_action - 1)
