from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.clean_up import constants as d, rewards as r

from marl_factory_grid.environment import constants as c


class CleanUp(Action):

    def __init__(self):
        super().__init__(d.CLEAN_UP)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if dirt := state[d.DIRT].by_pos(entity.pos):
            new_dirt_amount = dirt.amount - state[d.DIRT].clean_amount

            if new_dirt_amount <= 0:
                state[d.DIRT].delete_env_object(dirt)
            else:
                dirt.set_new_amount(max(new_dirt_amount, c.VALUE_FREE_CELL))
            valid = c.VALID
            print_str = f'{entity.name} did just clean up some dirt at {entity.pos}.'
            state.print(print_str)
            reward = r.CLEAN_UP_VALID
            identifier = d.CLEAN_UP
        else:
            valid = c.NOT_VALID
            print_str = f'{entity.name} just tried to clean up some dirt at {entity.pos}, but failed.'
            state.print(print_str)
            reward = r.CLEAN_UP_FAIL
            identifier = d.CLEAN_UP_FAIL

        return ActionResult(identifier=identifier, validity=valid, reward=reward, entity=entity)
