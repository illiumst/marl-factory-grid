from typing import List, Tuple, Union

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.groups.objects import _Objects
from marl_factory_grid.environment.entity.object import _Object
import marl_factory_grid.environment.constants as c


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

    def spawn(self, coords_or_quantity: Union[int, List[Tuple[(int, int)]]], *entity_args):  # woihn mit den args
        if isinstance(coords_or_quantity, int):
            self.add_items([self._entity() for _ in range(coords_or_quantity)])
        else:
            self.add_items([self._entity(pos) for pos in coords_or_quantity])
        return c.VALID

    def despawn(self, items: List[_Object]):
        items = [items] if isinstance(items, _Object) else items
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

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, x in enumerate(self) if x.belongs_to_entity(entity)))
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
            print()

    @property
    def positions(self):
        return [e.pos for e in self]

    def notify_del_entity(self, entity: Entity):
        try:
            self.pos_dict[entity.pos].remove(entity)
        except (ValueError, AttributeError):
            pass
