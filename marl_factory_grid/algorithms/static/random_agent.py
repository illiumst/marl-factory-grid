from random import randint

from marl_factory_grid.algorithms.static.TSP_base_agent import TSPBaseAgent

future_planning = 7


class TSPRandomAgent(TSPBaseAgent):

    def __init__(self, n_actions, *args, **kwargs):
        """
        Initializes a TSPRandomAgent that performs random actions from within his action space.

        :param n_actions: Number of possible actions.
        :type n_actions: int
        """
        super(TSPRandomAgent, self).__init__(*args, **kwargs)
        self.n_action = n_actions

    def predict(self, *_, **__):
        """
        Predicts the next action randomly.

        :return: Predicted action.
        :rtype: int
        """
        return randint(0, self.n_action - 1)
