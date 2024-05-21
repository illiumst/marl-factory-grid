import numpy as np

from marl_factory_grid.algorithms.static.TSP_base_agent import TSPBaseAgent

from marl_factory_grid.modules.items import constants as i
from marl_factory_grid.environment import constants as c

future_planning = 7
inventory_size  = 3

MODE_GET        = 'Mode_Get'
MODE_BRING      = 'Mode_Bring'


class TSPItemAgent(TSPBaseAgent):

    def __init__(self, *args, mode=MODE_GET, **kwargs):
        """
        Initializes a TSPItemAgent that colects items in the environment, stores them in his inventory and drops them off
        at a drop-off location.

        :param mode: Mode of the agent, either MODE_GET or MODE_BRING.
        """
        super(TSPItemAgent, self).__init__(*args, **kwargs)
        self.mode = mode
        self.fallback_action = c.NOOP

    def predict(self, *_, **__):
        item_at_position = self._env.state[i.ITEM].by_pos(self.state.pos)
        dropoff_at_position = self._env.state[i.DROP_OFF].by_pos(self.state.pos)
        if item_at_position:
            # Translate the action_object to an integer to have the same output as any other model
            action = i.ITEM_ACTION
        elif dropoff_at_position:
            # Translate the action_object to an integer to have the same output as any other model
            action = i.ITEM_ACTION
        elif door := self._door_is_close(self._env.state):
            action = self._use_door_or_move(door, i.DROP_OFF if self.mode == MODE_BRING else i.ITEM)
        else:
            action = self._choose()
        self.action_list.append(action)
        # Translate the action_object to an integer to have the same output as any other model
        try:
            action_obj = next(action_i for action_i, a in enumerate(self.state.actions) if a.name == action)
        except (StopIteration, UnboundLocalError):
            print('Will not happen')
            raise EnvironmentError
        # noinspection PyUnboundLocalVariable
        if self.mode == MODE_BRING and len(self._env[i.INVENTORY].by_entity(self.state)):
            pass
        elif self.mode == MODE_BRING and not len(self._env[i.INVENTORY].by_entity(self.state)):
            self.mode = MODE_GET
        elif self.mode == MODE_GET and len(self._env[i.INVENTORY].by_entity(self.state)) > inventory_size:
            self.mode = MODE_BRING
        else:
            pass
        return action_obj

    def _choose(self):
        """
        Internal Usage. Chooses the action based on the agent's mode and the environment state.

        :return: Chosen action.
        :rtype: int
        """
        target = i.DROP_OFF if self.mode == MODE_BRING else i.ITEM
        if len(self._env.state[i.ITEM]) >= 1:
            action = self._predict_move(target)

        elif len(self._env[i.INVENTORY].by_entity(self.state)):
            self.mode = MODE_BRING
            action = self._predict_move(target)
        else:
            action = int(np.random.randint(self._env.action_space.n))
        # noinspection PyUnboundLocalVariable
        return action
