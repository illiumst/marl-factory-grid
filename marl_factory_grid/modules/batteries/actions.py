from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.batteries import constants as b, rewards as r
from marl_factory_grid.environment import constants as c


class BtryCharge(Action):

    def __init__(self):
        super().__init__(b.CHARGE)

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
                            reward=r.CHARGE_VALID if valid else r.CHARGE_FAIL)
