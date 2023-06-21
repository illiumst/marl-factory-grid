from marl_factory_grid.algorithms.static.TSP_base_agent import TSPBaseAgent

from marl_factory_grid.modules.clean_up import constants as di

future_planning = 7


class TSPDirtAgent(TSPBaseAgent):

    def __init__(self, *args, **kwargs):
        super(TSPDirtAgent, self).__init__(*args, **kwargs)

    def predict(self, *_, **__):
        if self._env.state[di.DIRT].by_pos(self.state.pos) is not None:
            # Translate the action_object to an integer to have the same output as any other model
            action = di.CLEAN_UP
        elif door := self._door_is_close():
            action = self._use_door_or_move(door, di.DIRT)
        else:
            action = self._predict_move(di.DIRT)
        # Translate the action_object to an integer to have the same output as any other model
        try:
            action_obj = next(action_i for action_i, a in enumerate(self.state.actions) if a.name == action)
        except (StopIteration, UnboundLocalError):
            print('Will not happen')
            raise EnvironmentError
        return action_obj
