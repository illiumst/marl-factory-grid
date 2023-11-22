from typing import Union

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.actions import Action
from marl_factory_grid.modules.destinations import constants as d
from marl_factory_grid.utils.results import ActionResult


class DestAction(Action):

    def __init__(self):
        """
        Attempts to wait at destination.
        """
        super().__init__(d.DESTINATION, d.REWARD_WAIT_VALID, d.REWARD_WAIT_FAIL)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if destination := state[d.DESTINATION].by_pos(entity.pos):
            valid = destination.do_wait_action(entity)
            state.print(f'{entity.name} just waited at {entity.pos}')
        else:
            valid = c.NOT_VALID
            state.print(f'{entity.name} just tried to "do_wait_action" at {entity.pos} but failed')
        return self.get_result(valid, entity)
