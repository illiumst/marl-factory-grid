from marl_factory_grid.algorithms.tsp.TSP_base_agent import TSPBaseAgent

from marl_factory_grid.modules.destinations import constants as d
from marl_factory_grid.modules.doors import constants as do
from marl_factory_grid.environment import constants as c


future_planning = 7


class TSPTargetAgent(TSPBaseAgent):

    def __init__(self, *args, **kwargs):
        """
        Initializes a TSPTargetAgent that aims to reach destinations.
        """
        super(TSPTargetAgent, self).__init__(*args, **kwargs)
        self.fallback_action = c.NOOP

    def _handle_doors(self, state):
        """
        Internal Usage. Handles the doors in the environment.

        :param state: The current environment state.
        :type state: marl_factory_grid.utils.states.Gamestate
        :return: Closest door entity or None if no doors are close.
        :rtype: marl_factory_grid.environment.entity.object.Entity or None
        """

        try:
            return next(y for x in state.entities.neighboring_positions(self.state.pos)
                        for y in state.entities.pos_dict[x] if do.DOOR in y.name)
        except StopIteration:
            return None

    def predict(self, *_, **__):
        if door := self._door_is_close(self._env.state):
            action = self._use_door_or_move(door, d.DESTINATION)
        else:
            action = self._predict_move(d.DESTINATION)
        self.action_list.append(action)
        # Translate the action_object to an integer to have the same output as any other model
        try:
            action_obj = next(action_i for action_i, a in enumerate(self.state.actions) if a.name == action)
        except (StopIteration, UnboundLocalError):
            print('Will not happen')
        return action_obj
