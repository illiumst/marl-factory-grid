from typing import List, Tuple, Union, Dict

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.environment.entity.object import Object
import marl_factory_grid.environment.constants as c
from marl_factory_grid.utils.results import Result


class Collection(Objects):
    _entity = Object  # entity?
    symbol = None

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_is_blocking_pos(self):
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

    @property
    def encodings(self):
        return [x.encoding for x in self]

    @property
    def spawn_rule(self):
        """Prevent SpawnRule creation if Objects are spawned by map, Doors e.g."""
        if self.symbol:
            return None
        elif self._spawnrule:
            return self._spawnrule
        else:
            return {c.SPAWN_ENTITY_RULE: dict(collection=self, coords_or_quantity=self._coords_or_quantity)}

    def __init__(self, size, *args, coords_or_quantity: int = None, ignore_blocking=False,
                 spawnrule: Union[None, Dict[str, dict]] = None,
                 **kwargs):
        super(Collection, self).__init__(*args, **kwargs)
        self._coords_or_quantity = coords_or_quantity
        self.size = size
        self._spawnrule = spawnrule
        self._ignore_blocking = ignore_blocking

    def trigger_spawn(self, state, *entity_args, coords_or_quantity=None, ignore_blocking=False,  **entity_kwargs):
        coords_or_quantity = coords_or_quantity if coords_or_quantity else self._coords_or_quantity
        if self.var_has_position:
            if self.var_has_position and isinstance(coords_or_quantity, int):
                if ignore_blocking or self._ignore_blocking:
                    coords_or_quantity = state.entities.floorlist[:coords_or_quantity]
                else:
                    coords_or_quantity = state.get_n_random_free_positions(coords_or_quantity)
            self.spawn(coords_or_quantity, *entity_args,  **entity_kwargs)
            state.print(f'{len(coords_or_quantity)} new {self.name} have been spawned at {coords_or_quantity}')
            return Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=len(coords_or_quantity))
        else:
            if isinstance(coords_or_quantity, int):
                self.spawn(coords_or_quantity, *entity_args,  **entity_kwargs)
                state.print(f'{coords_or_quantity} new {self.name} have been spawned randomly.')
                return Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=coords_or_quantity)
            else:
                raise ValueError(f'{self._entity.__name__} has no position!')

    def spawn(self, coords_or_quantity: Union[int, List[Tuple[(int, int)]]], *entity_args, **entity_kwargs):
        if self.var_has_position:
            if isinstance(coords_or_quantity, int):
                raise ValueError(f'{self._entity.__name__} should have a position!')
            else:
                self.add_items([self._entity(pos, *entity_args, **entity_kwargs) for pos in coords_or_quantity])
        else:
            if isinstance(coords_or_quantity, int):
                self.add_items([self._entity(*entity_args, **entity_kwargs) for _ in range(coords_or_quantity)])
            else:
                raise ValueError(f'{self._entity.__name__} has no  position!')
        return c.VALID

    def despawn(self, items: List[Object]):
        items = [items] if isinstance(items, Object) else items
        for item in items:
            del self[item]

    def add_item(self, item: Entity):
        assert self.var_has_position or (len(self) <= self.size)
        super(Collection, self).add_item(item)
        return self

    def delete_env_object(self, env_object):
        del self[env_object.name]

    def delete_env_object_by_name(self, name):
        del self[name]

    @property
    def obs_pairs(self):
        pair_list = [(self.name, self)]
        try:
            if self.var_can_be_bound:
                pair_list.extend([(a.name, a) for a in self])
        except AttributeError:
            pass
        return pair_list

    def by_entity(self, entity):
        try:
            return next((x for x in self if x.belongs_to_entity(entity)))
        except (StopIteration, AttributeError):
            return None

    def render(self):
        if self.var_has_position:
            return [y for y in [x.render() for x in self] if y is not None]
        else:
            return []

    @classmethod
    def from_coordinates(cls, positions: [(int, int)], *args, entity_kwargs=None, **kwargs, ):
        collection = cls(*args, **kwargs)
        collection.add_items(
            [cls._entity(tuple(pos), **entity_kwargs if entity_kwargs is not None else {}) for pos in positions])
        return collection

    def __delitem__(self, name):
        idx, obj = next((i, obj) for i, obj in enumerate(self) if obj.name == name)
        try:
            for observer in obj.observers:
                observer.notify_del_entity(obj)
        except AttributeError:
            pass
        super().__delitem__(name)

    def by_pos(self, pos: (int, int)):
        pos = tuple(pos)
        try:
            return self.pos_dict[pos]
        except StopIteration:
            pass
        except ValueError:
            pass

    @property
    def positions(self):
        return [e.pos for e in self]

    def notify_del_entity(self, entity: Entity):
        try:
            self.pos_dict[entity.pos].remove(entity)
        except (ValueError, AttributeError):
            pass
