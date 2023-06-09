from typing import Union

from environment.actions import Action
from environment.utils.results import ActionResult

from modules.destinations import constants as d, rewards as r
from environment import constants as c


class DestAction(Action):

    def __init__(self):
        super().__init__(d.DESTINATION)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if destination := state[d.DESTINATION].by_pos(entity.pos):
            valid = destination.do_wait_action(entity)
            state.print(f'{entity.name} just waited at {entity.pos}')
        else:
            valid = c.NOT_VALID
            state.print(f'{entity.name} just tried to do_wait_action do_wait_action at {entity.pos} but failed')
        return ActionResult(entity=entity, identifier=self._identifier, validity=valid,
                            reward=r.WAIT_VALID if valid else r.WAIT_FAIL)
