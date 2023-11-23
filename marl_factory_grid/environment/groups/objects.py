from collections import defaultdict
from typing import List, Iterator, Union

import numpy as np

from marl_factory_grid.environment.entity.object import Object
import marl_factory_grid.environment.constants as c
from marl_factory_grid.utils import helpers as h


class Objects:
    _entity = Object

    @property
    def var_can_be_bound(self):
        return False

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
        pair_list = [(self.name, self)]
        pair_list.extend([(a.name, a) for a in self])
        return pair_list

    @property
    def names(self):
        # noinspection PyUnresolvedReferences
        return [x.name for x in self]

    @property
    def name(self):
        return f'{self.__class__.__name__}'

    def __init__(self, *args, **kwargs):
        self._data = defaultdict(lambda: None)
        self._observers = set(self)
        self.pos_dict = defaultdict(list)

    def __len__(self):
        return len(self._data)

    def __iter__(self) -> Iterator[Union[Object, None]]:
        return iter(self.values())

    def add_item(self, item: _entity):
        assert_str = f'All item names have to be of type {self._entity}, but were {item.__class__}.,'
        assert isinstance(item, self._entity), assert_str
        assert self._data[item.name] is None, f'{item.name} allready exists!!!'
        self._data.update({item.name: item})
        item.set_collection(self)
        if hasattr(self, "var_has_position") and self.var_has_position:
            item.add_observer(self)
        for observer in self.observers:
            observer.notify_add_entity(item)
        return self

    def remove_item(self, item: _entity):
        for observer in item.observers:
            observer.notify_del_entity(item)
        # noinspection PyTypeChecker
        del self._data[item.name]
        return True

    def __delitem__(self, name):
        return self.remove_item(self[name])

    # noinspection PyUnresolvedReferences
    def del_observer(self, observer):
        self.observers.remove(observer)
        for entity in self:
            if observer in entity.observers:
                entity.del_observer(observer)

    # noinspection PyUnresolvedReferences
    def add_observer(self, observer):
        self.observers.add(observer)
        for entity in self:
            entity.add_observer(observer)

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

    def by_name(self, name):
        return next(x for x in self if x.name == name)

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
        return f'{self.__class__.__name__}[{len(self)}]'

    def notify_del_entity(self, entity: Object):
        try:
            # noinspection PyUnresolvedReferences
            self.pos_dict[entity.pos].remove(entity)
        except (AttributeError, ValueError, IndexError):
            pass

    def notify_add_entity(self, entity: Object):
        try:
            if self not in entity.observers:
                entity.add_observer(self)
            if entity.var_has_position:
                if entity not in self.pos_dict[entity.pos]:
                    self.pos_dict[entity.pos].append(entity)
        except (ValueError, AttributeError):
            pass

    def summarize_states(self):
        # FIXME PROTOBUFF
        #  return [e.summarize_state() for e in self]
        return [e.summarize_state() for e in self]

    def by_entity(self, entity):
        try:
            return h.get_first(self, filter_by=lambda x: x.belongs_to_entity(entity))
        except (StopIteration, AttributeError):
            return None

    def idx_by_entity(self, entity):
        try:
            return h.get_first_index(self, filter_by=lambda x: x == entity)
        except (StopIteration, AttributeError):
            return None

    def reset(self):
        self._data = defaultdict(lambda: None)
        self._observers = set(self)
        self.pos_dict = defaultdict(list)

