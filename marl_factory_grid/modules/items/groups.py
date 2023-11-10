from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.environment.groups.mixins import IsBoundMixin
from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.modules.items import constants as i
from marl_factory_grid.modules.items.entitites import Item, DropOffLocation
from marl_factory_grid.utils.results import Result


class Items(Collection):
    _entity = Item

    @property
    def var_has_position(self):
        return True

    @property
    def is_blocking_light(self):
        return False

    @property
    def can_collide(self):
        return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trigger_spawn(self, state, *entity_args, coords_or_quantity=None, **entity_kwargs) -> [Result]:
        coords_or_quantity = coords_or_quantity if coords_or_quantity else self._coords_or_quantity
        assert coords_or_quantity

        if item_to_spawns := max(0, (coords_or_quantity - len(self))):
            return super().trigger_spawn(state,
                                         *entity_args,
                                         coords_or_quantity=item_to_spawns,
                                         **entity_kwargs)
        else:
            state.print('No Items are spawning, limit is reached.')
            return Result(identifier=f'{self.name}_spawn', validity=c.NOT_VALID, value=coords_or_quantity)


class Inventory(IsBoundMixin, Collection):
    _accepted_objects = Item

    @property
    def var_can_be_bound(self):
        return True

    @property
    def obs_tag(self):
        return self.name

    def __init__(self, agent: Agent, *args, **kwargs):
        super(Inventory, self).__init__(*args, **kwargs)
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


class Inventories(Objects):
    _entity = Inventory

    var_can_move = False
    var_has_position = False

    symbol = None

    @property
    def spawn_rule(self):
        return {c.SPAWN_ENTITY_RULE: dict(collection=self, coords_or_quantity=None)}

    def __init__(self, size: int, *args, **kwargs):
        super(Inventories, self).__init__(*args, **kwargs)
        self.size = size
        self._obs = None
        self._lazy_eval_transforms = []

    def spawn(self, agents, *args, **kwargs):
        self.add_items([self._entity(agent, self.size, *args, **kwargs) for _, agent in enumerate(agents)])
        return [Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=len(self))]

    def trigger_spawn(self, state, *args, **kwargs) -> [Result]:
        return self.spawn(state[c.AGENT], *args, **kwargs)

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


class DropOffLocations(Collection):
    _entity = DropOffLocation

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_move(self):
        return False

    @property
    def var_has_position(self):
        return True

    def __init__(self, *args, **kwargs):
        super(DropOffLocations, self).__init__(*args, **kwargs)

    @staticmethod
    def trigger_drop_off_location_spawn(state, n_locations):
        empty_positions = state.entities.empty_positions[:n_locations]
        do_entites = state[i.DROP_OFF]
        drop_offs = [DropOffLocation(pos) for pos in empty_positions]
        do_entites.add_items(drop_offs)
