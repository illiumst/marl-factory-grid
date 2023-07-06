from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.modules.doors import constants as d, rewards as r
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import ActionResult


class DoorUse(Action):

    def __init__(self):
        super().__init__(d.ACTION_DOOR_USE)

    def do(self, entity, state) -> Union[None, ActionResult]:
        # Check if agent really is standing on a door:
        e = state.entities.get_near_pos(entity.pos)
        try:
            door = next(x for x in e if x.name.startswith(d.DOOR))
            valid = door.use()
            state.print(f'{entity.name} just used a {door.name} at {door.pos}')
            return ActionResult(entity=entity, identifier=self._identifier, validity=valid, reward=r.USE_DOOR_VALID)

        except StopIteration:
            # When he doesn't...
            state.print(f'{entity.name} just tried to use a door at {entity.pos}, but there is none.')
            return ActionResult(entity=entity, identifier=self._identifier,
                                validity=c.NOT_VALID, reward=r.USE_DOOR_FAIL)
