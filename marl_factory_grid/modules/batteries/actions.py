from typing import Union

import marl_factory_grid.modules.batteries.constants
from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.batteries import constants as b
from marl_factory_grid.environment import constants as c


class BtryCharge(Action):

    def __init__(self):
        super().__init__(b.ACTION_CHARGE)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if charge_pod := state[b.CHARGE_PODS].by_pos(entity.pos):
            valid = charge_pod.charge_battery(state[b.BATTERIES].by_entity(entity))
            if valid:
                state.print(f'{entity.name} just charged batteries at {charge_pod.name}.')
            else:
                state.print(f'{entity.name} failed to charged batteries at {charge_pod.name}.')
        else:
            valid = c.NOT_VALID
            state.print(f'{entity.name} failed to charged batteries at {entity.pos}.')
        return ActionResult(entity=entity, identifier=self._identifier, validity=valid,
                            reward=marl_factory_grid.modules.batteries.constants.REWARD_CHARGE_VALID if valid else marl_factory_grid.modules.batteries.constants.Reward_CHARGE_FAIL)
