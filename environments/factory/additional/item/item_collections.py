from typing import List

import numpy as np

from environments.factory.base.objects import Floor, Agent
from environments.factory.base.registers import EntityCollection, BoundEnvObjCollection, ObjectCollection
from environments.factory.additional.item.item_entities import Item, DropOffLocation


class ItemRegister(EntityCollection):

    _accepted_objects = Item

    def spawn_items(self, tiles: List[Floor]):
        items = [Item(tile, self) for tile in tiles]
        self.add_additional_items(items)

    def despawn_items(self, items: List[Item]):
        items = [items] if isinstance(items, Item) else items
        for item in items:
            del self[item]


class Inventory(BoundEnvObjCollection):

    @property
    def name(self):
        return f'{self.__class__.__name__}({self._bound_entity.name})'

    def __init__(self, agent: Agent, capacity: int, *args, **kwargs):
        super(Inventory, self).__init__(agent, *args,  is_blocking_light=False, can_be_shadowed=False,  **kwargs)
        self.capacity = capacity

    def as_array(self):
        if self._array is None:
            self._array = np.zeros((1, *self._shape))
        return super(Inventory, self).as_array()

    def summarize_states(self, **kwargs):
        attr_dict = {key: str(val) for key, val in self.__dict__.items() if not key.startswith('_') and key != 'data'}
        attr_dict.update(dict(items={key: val.summarize_state(**kwargs) for key, val in self.items()}))
        attr_dict.update(dict(name=self.name))
        return attr_dict

    def pop(self):
        item_to_pop = self[0]
        self.delete_env_object(item_to_pop)
        return item_to_pop


class Inventories(ObjectCollection):

    _accepted_objects = Inventory
    is_blocking_light = False
    can_be_shadowed = False

    def __init__(self, obs_shape, *args, **kwargs):
        super(Inventories, self).__init__(*args, is_per_agent=True, individual_slices=True, **kwargs)
        self._obs_shape = obs_shape

    def as_array(self):
        return np.stack([inventory.as_array() for inv_idx, inventory in enumerate(self)])

    def spawn_inventories(self, agents, capacity):
        inventories = [self._accepted_objects(agent, capacity, self._obs_shape)
                       for _, agent in enumerate(agents)]
        self.add_additional_items(inventories)

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, inv in enumerate(self) if inv.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def by_entity(self, entity):
        try:
            return next((inv for inv in self if inv.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def summarize_states(self, **kwargs):
        return {key: val.summarize_states(**kwargs) for key, val in self.items()}


class DropOffLocations(EntityCollection):

    _accepted_objects = DropOffLocation
