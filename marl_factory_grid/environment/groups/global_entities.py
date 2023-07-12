from collections import defaultdict
from operator import itemgetter
from typing import Dict

from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.utils.helpers import POS_MASK


class Entities(Objects):
    _entity = Objects

    @staticmethod
    def neighboring_positions(pos):
        return (POS_MASK + pos).reshape(-1, 2)

    def get_near_pos(self, pos):
        return [y for x in itemgetter(*(tuple(x) for x in self.neighboring_positions(pos)))(self.pos_dict) for y in x]

    def render(self):
        return [y for x in self for y in x.render() if x is not None]

    @property
    def names(self):
        return list(self._data.keys())

    def __init__(self):
        self.pos_dict = defaultdict(list)
        super().__init__()

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
        self[name]._observers.delete(self)
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
        # found_entities = [y for y in (x.by_pos(pos) for x in self.values() if hasattr(x, 'by_pos')) if y is not None]
        # return found_entities

    @property
    def positions(self):
        return [k for k, v in self.pos_dict.items() for _ in v]
