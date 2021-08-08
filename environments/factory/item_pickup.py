import time
from collections import deque
from typing import List, Union, NamedTuple
import numpy as np

from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Object, Slice, Entity
from environments.factory.base.registers import Entities

from environments.factory.renderer import Renderer
from environments.utility_classes import MovementProperties



ITEM = 'item'
INVENTORY = 'inventory'
PICK_UP = 'pick_up'
DROP_DOWN = 'drop_down'
ITEM_ACTION = 'item_action'
NO_ITEM = 0
ITEM_DROP_OFF = -1


def inventory_slice_name(agent):
    return f'{agent.name}_{INVENTORY}'


class DropOffLocation(Entity):

    def __init__(self, *args, storage_size_until_full: int = 5, **kwargs):
        super(DropOffLocation, self).__init__(*args, **kwargs)
        self.storage = deque(maxlen=storage_size_until_full)

    def place_item(self, item):
        self.storage.append(item)
        return True

    @property
    def is_full(self):
        return self.storage.maxlen == len(self.storage)


class ItemProperties(NamedTuple):
    n_items: int = 1                    # How many items are there at the same time
    spawn_frequency: int = 5            # Spawn Frequency in Steps
    max_dropoff_storage_size: int = 5   # How many items are needed until the drop off is full
    max_agent_storage_size: int = 5     # How many items are needed until the agent inventory is full


# noinspection PyAttributeOutsideInit
class ItemFactory(BaseFactory):
    def __init__(self, item_properties: ItemProperties, *args, **kwargs):
        self.item_properties = item_properties
        self._item_rng = np.random.default_rng(kwargs.get('seed', default=time.time_ns()))
        super(ItemFactory, self).__init__(*args, **kwargs)

    @property
    def additional_actions(self) -> Union[str, List[str]]:
        return [ITEM_ACTION]

    @property
    def additional_entities(self) -> Union[Entities, List[Entities]]:
        return []

    @property
    def additional_slices(self) -> Union[Slice, List[Slice]]:
        return [Slice(ITEM, np.zeros(self._level_shape))] + [
            Slice(inventory_slice_name(agent), np.zeros(self._level_shape)) for agent in self._agents]

    def _is_item_action(self, action):
        if isinstance(action, str):
            action = self._actions.by_name(action)
        return self._actions[action].name == ITEM_ACTION

    def do_item_action(self, agent):
        item_slice = self._slices.by_name(ITEM).slice
        inventory_slice = self._slices.by_name(inventory_slice_name(agent)).slice

        if item := item_slice[agent.pos]:
            if item == ITEM_DROP_OFF:

                valid = self._item_drop_off.place_item(inventory_slice.sum())


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

    def spawn_drop_off_location(self):
        single_empty_tile = self._tiles.empty_tiles[0]
        self._item_drop_off = DropOffLocation(storage_size_until_full=self.item_properties.max_dropoff_storage_size)

    def calculate_reward(self) -> (int, dict):
        pass

    def render(self, mode='human'):
        pass


