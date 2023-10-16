from collections import defaultdict
from typing import List

import numpy as np

from marl_factory_grid.environment.entity.object import Object
import marl_factory_grid.environment.constants as c


class Objects:
    _entity = Object

    @property
    def observers(self):
        return self._observers

    @property
    def obs_tag(self):
        return self.__class__.__name__

    @staticmethod
    def render():
        return []

    @property
    def obs_pairs(self):
        return [(self.name, self)]

    @property
    def names(self):
        # noinspection PyUnresolvedReferences
        return [x.name for x in self]

    @property
    def name(self):
        return f'{self.__class__.__name__}'

    def __init__(self, *args, **kwargs):
        self._data = defaultdict(lambda: None)
        self._observers = [self]
        self.pos_dict = defaultdict(list)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self.values())

    def add_item(self, item: _entity):
        assert_str = f'All item names have to be of type {self._entity}, but were {item.__class__}.,'
        assert isinstance(item, self._entity), assert_str
        assert self._data[item.name] is None, f'{item.name} allready exists!!!'
        self._data.update({item.name: item})
        item.set_collection(self)
        # self.notify_add_entity(item)
        for observer in self.observers:
            observer.notify_add_entity(item)
        return self

    # noinspection PyUnresolvedReferences
    def del_observer(self, observer):
        self.observers.remove(observer)
        for entity in self:
            if observer in entity.observers:
                entity.del_observer(observer)

    # noinspection PyUnresolvedReferences
    def add_observer(self, observer):
        self.observers.append(observer)
        for entity in self:
            if observer not in entity.observers:
                entity.add_observer(observer)

    def __delitem__(self, name):
        for observer in self.observers:
            observer.notify_del_entity(name)
        # noinspection PyTypeChecker
        del self._data[name]

    def add_items(self, items: List[_entity]):
        for item in items:
            self.add_item(item)
        return self

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def _get_index(self, item):
        try:
            return next(i for i, v in enumerate(self._data.values()) if v == item)
        except StopIteration:
            return None

    def __getitem__(self, item):
        if isinstance(item, (int, np.int64, np.int32)):
            if item < 0:
                item = len(self._data) - abs(item)
            try:
                return next(v for i, v in enumerate(self._data.values()) if i == item)
            except StopIteration:
                return None
        try:
            return self._data[item]
        except KeyError:
            return None
        except TypeError:
            print('Ups')
            raise TypeError

    def __repr__(self):
        return f'{self.__class__.__name__}[{dict(self._data)}]'

    def spawn(self, n: int):
        self.add_items([self._entity() for _ in range(n)])
        return c.VALID

    def despawn(self, items: List[Object]):
        items = [items] if isinstance(items, Object) else items
        for item in items:
            del self[item]

    def notify_change_pos(self, entity: object):
        try:
            self.pos_dict[entity.last_pos].remove(entity)
        except (ValueError, AttributeError):
            pass
        if entity.var_has_position:
            try:
                self.pos_dict[entity.pos].append(entity)
            except (ValueError, AttributeError):
                pass

    def notify_del_entity(self, entity: Object):
        try:
            self.pos_dict[entity.pos].remove(entity)
        except (ValueError, AttributeError):
            pass

    def notify_add_entity(self, entity: Object):
        try:
            if self not in entity.observers:
                entity.add_observer(self)
            self.pos_dict[entity.pos].append(entity)
        except (ValueError, AttributeError):
            pass

    def summarize_states(self):
        # FIXME PROTOBUFF
        #  return [e.summarize_state() for e in self]
        return [e.summarize_state() for e in self]
