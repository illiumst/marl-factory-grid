from marl_factory_grid.algorithms.static.TSP_base_agent import TSPBaseAgent

from marl_factory_grid.modules.coins import constants as c
from marl_factory_grid.environment import constants as e

future_planning = 7


class TSPCoinAgent(TSPBaseAgent):

    def __init__(self, *args, **kwargs):
        """
        Initializes a TSPCoinAgent that aims to collect coins in the environment.
        """
        super(TSPCoinAgent, self).__init__(*args, **kwargs)
        self.fallback_action = e.NOOP

    def predict(self, *_, **__):
        """
        Predicts the next action based on the presence of coins in the environment.

        :return: Predicted action.
        :rtype: int
        """
        coin_at_position = self._env.state[c.COIN].by_pos(self.state.pos)
        if coin_at_position:
            # Translate the action_object to an integer to have the same output as any other model
            action = c.COLLECT
        elif door := self._door_is_close(self._env.state):
            action = self._use_door_or_move(door, c.COIN)
        else:
            action = self._predict_move(c.COIN)
        self.action_list.append(action)
        # Translate the action_object to an integer to have the same output as any other model
        try:
            action_obj = next(action_i for action_i, a in enumerate(self.state.actions) if a.name == action)
        except (StopIteration, UnboundLocalError):
            print('Will not happen')
            raise EnvironmentError
        return action_obj
