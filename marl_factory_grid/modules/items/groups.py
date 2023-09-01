from typing import List

from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.environment.groups.mixins import PositionMixin, IsBoundMixin, HasBoundMixin
from marl_factory_grid.environment.entity.wall_floor import Floor
from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.modules.items.entitites import Item, DropOffLocation


class Items(PositionMixin, EnvObjects):

    _entity = Item
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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


class Inventories(HasBoundMixin, Objects):

    _entity = Inventory
    var_can_move = False

    def __init__(self, size: int, *args, **kwargs):
        super(Inventories, self).__init__(*args, **kwargs)
        self.size = size
        self._obs = None
        self._lazy_eval_transforms = []

    def spawn(self, agents):
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
