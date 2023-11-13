from collections import defaultdict
from operator import itemgetter
from random import shuffle
from typing import Dict

from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.utils.helpers import POS_MASK


class Entities(Objects):
    _entity = Objects

    @staticmethod
    def neighboring_positions(pos):
        return [tuple(x) for x in (POS_MASK + pos).reshape(-1, 2)]

    def get_entities_near_pos(self, pos):
        return [y for x in itemgetter(*self.neighboring_positions(pos))(self.pos_dict) for y in x]

    def render(self):
        return [y for x in self for y in x.render() if x is not None]

    @property
    def names(self):
        return list(self._data.keys())

    @property
    def floorlist(self):
        shuffle(self._floor_positions)
        return [x for x in self._floor_positions]

    def __init__(self, floor_positions):
        self._floor_positions = floor_positions
        self.pos_dict = None
        super().__init__()

    def __repr__(self):
        return f'{self.__class__.__name__}{[x for x in self]}'

    def guests_that_can_collide(self, pos):
        return [x for val in self.pos_dict[pos] for x in val if x.var_can_collide]

    @property
    def empty_positions(self):
        empty_positions = [key for key in self.floorlist if not self.pos_dict[key]]
        shuffle(empty_positions)
        return empty_positions

    @property
    def occupied_positions(self):   # positions that are not empty
        empty_positions = [key for key in self.floorlist if self.pos_dict[key]]
        shuffle(empty_positions)
        return empty_positions

    @property
    def blocked_positions(self):
        blocked_positions = [key for key, val in self.pos_dict.items() if any([x.var_is_blocking_pos for x in val])]
        shuffle(blocked_positions)
        return blocked_positions

    @property
    def free_positions_generator(self):
        generator = (
            key for key in self.floorlist if all(not x.var_can_collide and not x.var_is_blocking_pos
                                                 for x in self.pos_dict[key])
                     )
        return generator

    @property
    def free_positions_list(self):
        return [x for x in self.free_positions_generator]

    def iter_entities(self):
        return iter((x for sublist in self.values() for x in sublist))

    def add_items(self, items: Dict):
        return self.add_item(items)

    def add_item(self, item: dict):
        assert_str = 'This group of entity has already been added!'
        assert not any([key for key in item.keys() if key in self.keys()]), assert_str
        self._data.update(item)
        for val in item.values():
            val.add_observer(self)
        return self

    def __contains__(self, item):
        return item in self._data

    def __delitem__(self, name):
        assert_str = 'This group of entity does not exist in this collection!'
        assert any([key for key in name.keys() if key in self.keys()]), assert_str
        self[name].del_observer(self)
        for entity in self[name]:
            entity.del_observer(self)
        return super(Entities, self).__delitem__(name)

    @property
    def obs_pairs(self):
        try:
            return [y for x in self for y in x.obs_pairs]
        except AttributeError:
            print('OhOh (debug me)')

    def by_pos(self, pos: (int, int)):
        return self.pos_dict[pos]

    @property
    def positions(self):
        return [k for k, v in self.pos_dict.items() for _ in v]

    def is_occupied(self, pos):
        return len([x for x in self.pos_dict[pos] if x.var_can_collide or x.var_is_blocking_pos]) >= 1

    def reset(self):
        self._observers = set(self)
        self.pos_dict = defaultdict(list)
        for entity_group in self:
            entity_group.reset()

            if hasattr(entity_group, "var_has_position") and entity_group.var_has_position:
                entity_group.add_observer(self)
