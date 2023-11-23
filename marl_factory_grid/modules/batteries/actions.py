from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.batteries import constants as b
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils import helpers as h


class Charge(Action):

    def __init__(self):
        """
        Checks if a charge pod is present at the entity's position.
        If found, it attempts to charge the battery using the charge pod.
        """
        super().__init__(b.ACTION_CHARGE, b.REWARD_CHARGE_VALID, b.Reward_CHARGE_FAIL)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if charge_pod := h.get_first(state[b.CHARGE_PODS].by_pos(entity.pos)):
            valid = h.get_first(charge_pod.charge_battery(entity, state))
            if valid:
                state.print(f'{entity.name} just charged batteries at {charge_pod.name}.')
            else:
                state.print(f'{entity.name} failed to charged batteries at {charge_pod.name}.')
        else:
            valid = c.NOT_VALID
            state.print(f'{entity.name} failed to charged batteries at {entity.pos}.')

        return self.get_result(valid, entity)
