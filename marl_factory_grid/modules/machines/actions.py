from typing import Union

import marl_factory_grid.modules.machines.constants
from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.machines import constants as m
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils import helpers as h


class MachineAction(Action):

    def __init__(self):
        super().__init__(m.MACHINE_ACTION, m.MAINTAIN_VALID, m.MAINTAIN_FAIL)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if machine := h.get_first(state[m.MACHINES].by_pos(entity.pos)):
            valid = machine.maintain()
            return self.get_result(valid, entity)

        else:
            return self.get_result(c.NOT_VALID, entity)
