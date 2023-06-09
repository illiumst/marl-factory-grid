from typing import List

from environment.groups.env_objects import EnvObjects
from environment.groups.objects import Objects
from environment.groups.mixins import PositionMixin, IsBoundMixin, HasBoundedMixin
from environment.entity.wall_floor import Floor
from environment.entity.agent import Agent
from modules.items.entitites import Item, DropOffLocation


class Items(PositionMixin, EnvObjects):

    _entity = Item
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def spawn_items(self, tiles: List[Floor]):
        items = [self._entity(tile) for tile in tiles]
        self.add_items(items)

    def despawn_items(self, items: List[Item]):
        items = [items] if isinstance(items, Item) else items
        for item in items:
            del self[item]


class Inventory(IsBoundMixin, EnvObjects):

    _accepted_objects = Item

    @property
    def obs_tag(self):
        return self.name

    def __init__(self, agent: Agent, *args, **kwargs):
        super(Inventory, self).__init__(*args,  **kwargs)
        self._collection = None
        self.bind(agent)

    def summarize_states(self, **kwargs):
        attr_dict = {key: val for key, val in self.__dict__.items() if not key.startswith('_') and key != 'data'}
        attr_dict.update(dict(items=[val.summarize_state(**kwargs) for key, val in self.items()]))
        attr_dict.update(dict(name=self.name, belongs_to=self._bound_entity.name))
        return attr_dict

    def pop(self):
        item_to_pop = self[0]
        self.delete_env_object(item_to_pop)
        return item_to_pop

    def set_collection(self, collection):
        self._collection = collection


class Inventories(HasBoundedMixin, Objects):

    _entity = Inventory
    can_move = False

    @property
    def obs_pairs(self):
        return [(x.name, x) for x in self]

    def __init__(self, size, *args, **kwargs):
        super(Inventories, self).__init__(*args, **kwargs)
        self.size = size
        self._obs = None
        self._lazy_eval_transforms = []

    def spawn_inventories(self, agents):
        inventories = [self._entity(agent, self.size,)
                       for _, agent in enumerate(agents)]
        self.add_items(inventories)

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
        return [val.summarize_states(**kwargs) for key, val in self.items()]


class DropOffLocations(PositionMixin, EnvObjects):

    _entity = DropOffLocation
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super(DropOffLocations, self).__init__(*args, **kwargs)
