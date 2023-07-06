from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.machines import constants as m, rewards as r
from marl_factory_grid.environment import constants as c


class MachineAction(Action):

    def __init__(self):
        super().__init__(m.MACHINE_ACTION)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if machine := state[m.MACHINES].by_pos(entity.pos):
            if valid := machine.maintain():
                return ActionResult(entity=entity, identifier=self._identifier, validity=valid, reward=r.MAINTAIN_VALID)
            else:
                return ActionResult(entity=entity, identifier=self._identifier, validity=valid, reward=r.MAINTAIN_FAIL)
        else:
            return ActionResult(entity=entity, identifier=self._identifier, validity=c.NOT_VALID, reward=r.MAINTAIN_FAIL)



