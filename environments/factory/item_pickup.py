from typing import List, Union, NamedTuple
import random

import numpy as np

from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Object, Slice
from environments.factory.base.registers import Entities

from environments.factory.renderer import Renderer, Entity
from environments.utility_classes import MovementProperties



ITEM = 'item'
INVENTORY = 'inventory'
PICK_UP = 'pick_up'
DROP_DOWN = 'drop_down'
ITEM_ACTION = 'item_action'
NO_ITEM = 0
ITEM_DROP_OFF = -1


class ItemProperties(NamedTuple):
    n_items: int = 1                    # How much does the robot clean with one actions.
    spawn_frequency: int = 5            # Spawn Frequency in Steps


# noinspection PyAttributeOutsideInit
class ItemFactory(BaseFactory):
    def __init__(self, item_properties: ItemProperties, *args, **kwargs):
        super(ItemFactory, self).__init__(*args, **kwargs)
        self.item_properties = item_properties

    @property
    def additional_actions(self) -> Union[str, List[str]]:
        return [ITEM_ACTION]

    @property
    def additional_entities(self) -> Union[Entities, List[Entities]]:
        return []

    @property
    def additional_slices(self) -> Union[Slice, List[Slice]]:
        return [Slice(ITEM, np.zeros(self._level_shape)), Slice(INVENTORY, np.zeros(self._level_shape))]

    def _is_item_action(self, action):
        if isinstance(action, str):
            action = self._actions.by_name(action)
        return self._actions[action].name == ITEM_ACTION

    def do_item_action(self, agent):
        item_slice = self._slices.by_name(ITEM).slice
        if item := item_slice[agent.pos]:
            if item == ITEM_DROP_OFF:

            self._slices.by_name(INVENTORY).slice[agent.pos] = item
            item_slice[agent.pos] = NO_ITEM
            return True
        else:
            return False

    def do_additional_actions(self, agent: Agent, action: int) -> bool:
        if self._is_item_action(action):
            valid = self.do_item_action(agent)
            return valid
        else:
            raise RuntimeError('This should not happen!!!')

    def do_additional_reset(self) -> None:
        self.spawn_drop_off_location()
        self.spawn_items(self.n_items)
        if self.n_items > 1:
            self._next_item_spawn = self.item_properties.spawn_frequency

    def calculate_reward(self) -> (int, dict):
        pass

    def render(self, mode='human'):
        pass


