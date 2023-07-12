from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.destinations import constants as d, rewards as r
from marl_factory_grid.environment import constants as c


class DestAction(Action):

    def __init__(self):
        super().__init__(d.DESTINATION)

    def do(self, entity, state) -> Union[None, ActionResult]:
        dest_entities = d.DESTINATION if d.DESTINATION in state else d.BOUNDDESTINATION
        assert dest_entities
        if destination := state[dest_entities].by_pos(entity.pos):
            valid = destination.do_wait_action(entity)
            state.print(f'{entity.name} just waited at {entity.pos}')
        else:
            valid = c.NOT_VALID
            state.print(f'{entity.name} just tried to do_wait_action do_wait_action at {entity.pos} but failed')
        return ActionResult(entity=entity, identifier=self._identifier, validity=valid,
                            reward=r.WAIT_VALID if valid else r.WAIT_FAIL)
