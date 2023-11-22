from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.modules.doors import constants as d
from marl_factory_grid.modules.doors.entitites import Door
from marl_factory_grid.utils.results import ActionResult


class DoorUse(Action):

    def __init__(self, **kwargs):
        """
        Attempts to interact with door (open/close it) and returns an action result if successful.
        """
        super().__init__(d.ACTION_DOOR_USE, d.REWARD_USE_DOOR_VALID, d.REWARD_USE_DOOR_FAIL, **kwargs)

    def do(self, entity, state) -> Union[None, ActionResult]:
        # Check if agent really is standing on a door:
        entities_close = state.entities.get_entities_near_pos(entity.pos)

        valid = False
        for door in [e for e in entities_close if isinstance(e, Door)]:
            try:
                # Will always be true, when there is at least a single door.
                valid = door.use()
                state.print(f'{entity.name} just used a {door.name} at {door.pos}')

            except AttributeError:
                pass
        if not valid:
            # When he doesn't stand next to a door tell me.
            state.print(f'{entity.name} just tried to use a door at {entity.pos}, but there is none.')
        return self.get_result(valid, entity)
