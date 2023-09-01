from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.items import constants as i, rewards as r
from marl_factory_grid.environment import constants as c


class ItemAction(Action):

    def __init__(self):
        super().__init__(i.ITEM_ACTION)

    def do(self, entity, state) -> Union[None, ActionResult]:
        inventory = state[i.INVENTORY].by_entity(entity)
        if drop_off := state[i.DROP_OFF].by_pos(entity.pos):
            if inventory:
                valid = drop_off.place_item(inventory.pop())
            else:
                valid = c.NOT_VALID
            if valid:
                state.print(f'{entity.name} just dropped of an item at {drop_off.pos}.')
            else:
                state.print(f'{entity.name} just tried to drop off at {entity.pos}, but failed.')
            reward = r.DROP_OFF_VALID if valid else r.DROP_OFF_FAIL
            return ActionResult(entity=entity, identifier=self._identifier, validity=valid, reward=reward)

        elif items := state[i.ITEM].by_pos(entity.pos):
            item = items[0]
            item.change_parent_collection(inventory)
            item.set_tile_to(state.NO_POS_TILE)
            state.print(f'{entity.name} just picked up an item at {entity.pos}')
            return ActionResult(entity=entity, identifier=self._identifier, validity=c.VALID, reward=r.PICK_UP_VALID)

        else:
            state.print(f'{entity.name} just tried to pick up an item at {entity.pos}, but failed.')
            return ActionResult(entity=entity, identifier=self._identifier, validity=c.NOT_VALID, reward=r.PICK_UP_FAIL)
