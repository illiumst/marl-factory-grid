from typing import List, Tuple

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.groups.objects import _Objects
from marl_factory_grid.environment.entity.object import _Object


class Collection(_Objects):
    _entity = _Object  # entity?

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
        return False

    # @property
    # def var_has_bound(self):
    #     return False  # batteries, globalpos, inventories true

    @property
    def var_can_be_bound(self):
        return False

    @property
    def encodings(self):
        return [x.encoding for x in self]

    def __init__(self, size, *args, **kwargs):
        super(Collection, self).__init__(*args, **kwargs)
        self.size = size

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
        return [(x.name, x) for x in self]

    def by_entity(self, entity):
        try:
            return next((x for x in self if x.belongs_to_entity(entity)))
        except (StopIteration, AttributeError):
            return None

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, x in enumerate(self) if x.belongs_to_entity(entity)))
        except (StopIteration, AttributeError):
            return None

    def spawn(self, coords: List[Tuple[(int, int)]]):
        self.add_items([self._entity(pos) for pos in coords])

    def render(self):
        return [y for y in [x.render() for x in self] if y is not None]

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
            print()

    @property
    def positions(self):
        return [e.pos for e in self]

    def notify_del_entity(self, entity: Entity):
        try:
            self.pos_dict[entity.pos].remove(entity)
        except (ValueError, AttributeError):
            pass
