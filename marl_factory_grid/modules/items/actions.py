from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.items import constants as i, rewards as r
from marl_factory_grid.environment import constants as c


class ItemAction(Action):

    def __init__(self, failed_dropoff_reward: float | None = None, valid_dropoff_reward: float | None = None, **kwargs):
        super().__init__(i.ITEM_ACTION, r.PICK_UP_FAIL, r.PICK_UP_VALID, **kwargs)
        self.failed_drop_off_reward = failed_dropoff_reward if failed_dropoff_reward is not None else r.DROP_OFF_FAIL
        self.valid_drop_off_reward = valid_dropoff_reward if valid_dropoff_reward is not None else r.DROP_OFF_FAIL

    def get_dropoff_result(self, validity, entity):
        reward = self.valid_drop_off_reward if validity else self.failed_drop_off_reward
        return ActionResult(self.__name__, validity, reward=reward, entity=entity)

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
            return self.get_dropoff_result(valid, entity)

        elif items := state[i.ITEM].by_pos(entity.pos):
            item = items[0]
            item.change_parent_collection(inventory)
            item.set_pos(c.VALUE_NO_POS)
            state.print(f'{entity.name} just picked up an item at {entity.pos}')
            return self.get_result(c.VALID, entity)

        else:
            state.print(f'{entity.name} just tried to pick up an item at {entity.pos}, but failed.')
            return self.get_result(c.NOT_VALID, entity)
