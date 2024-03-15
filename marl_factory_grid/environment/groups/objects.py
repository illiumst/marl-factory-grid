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
        """
        Property indicating whether objects in the collection can be bound to another entity.
        """
        return False

    @property
    def observers(self):
        """
        Property returning a set of observers associated with the collection.
        """
        return self._observers

    @property
    def obs_tag(self):
        """
        Property providing a tag for observation purposes.
        """
        return self.__class__.__name__

    @staticmethod
    def render():
        """
        Static method returning an empty list. Override this method in derived classes for rendering functionality.
        """
        return []

    @property
    def obs_pairs(self):
        """
        Property returning a list of pairs containing the names and corresponding objects within the collection.
        """
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
        """
        Returns the number of objects in the collection.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[Union[Object, None]]:
        return iter(self.values())

    def add_item(self, item: _entity):
        """
         Adds an item to the collection.


        :param item: The object to add to the collection.

        :returns: The updated collection.

        Raises:
            AssertionError: If the item is not of the correct type or already exists in the collection.
        """
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
        """
        Removes an item from the collection.
        """
        for observer in item.observers:
            observer.notify_del_entity(item)
        # noinspection PyTypeChecker
        del self._data[item.name]
        return True

    def __delitem__(self, name):
        return self.remove_item(self[name])

    # noinspection PyUnresolvedReferences
    def del_observer(self, observer):
        """
        Removes an observer from the collection and its entities.
        """
        self.observers.remove(observer)
        for entity in self:
            if observer in entity.observers:
                entity.del_observer(observer)

    # noinspection PyUnresolvedReferences
    def add_observer(self, observer):
        """
        Adds an observer to the collection and its entities.
        """
        self.observers.add(observer)
        for entity in self:
            entity.add_observer(observer)

    def add_items(self, items: List[_entity]):
        """
        Adds a list of items to the collection.

        :param items: List of items to add.
        :type items: List[_entity]
        :returns: The updated collection.
        """
        for item in items:
            self.add_item(item)
        return self

    def keys(self):
        """
        Returns the keys (names) of the objects in the collection.
        """
        return self._data.keys()

    def values(self):
        """
        Returns the values (objects) in the collection.
        """
        return self._data.values()

    def items(self):
        """
        Returns the items (name-object pairs) in the collection.
        """
        return self._data.items()

    def _get_index(self, item):
        """
        Gets the index of an item in the collection.
        """
        try:
            return next(i for i, v in enumerate(self._data.values()) if v == item)
        except StopIteration:
            return None

    def by_name(self, name):
        """
        Gets an object from the collection by its name.
        """
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
        """
        Notifies the collection that an entity has been deleted.
        """
        try:
            # noinspection PyUnresolvedReferences
            self.pos_dict[entity.pos].remove(entity)
        except (AttributeError, ValueError, IndexError):
            pass

    def notify_add_entity(self, entity: Object):
        """
        Notifies the collection that an entity has been added.
        """
        try:
            if self not in entity.observers:
                entity.add_observer(self)
            if entity.var_has_position:
                if entity not in self.pos_dict[entity.pos]:
                    self.pos_dict[entity.pos].append(entity)
        except (ValueError, AttributeError):
            pass

    def summarize_states(self):
        """
        Summarizes the states of all entities in the collection.

        :returns: A list of dictionaries representing the summarized states of the entities.
        :rtype: List[dict]
        """
        # FIXME PROTOBUFF
        #  return [e.summarize_state() for e in self]
        return [e.summarize_state() for e in self]

    def by_entity(self, entity):
        """
        Gets an entity from the collection that belongs to a specified entity.
        """
        try:
            return h.get_first(self, filter_by=lambda x: x.belongs_to_entity(entity))
        except (StopIteration, AttributeError):
            return None

    def idx_by_entity(self, entity):
        """
        Gets the index of an entity in the collection.
        """
        try:
            return h.get_first_index(self, filter_by=lambda x: x == entity)
        except (StopIteration, AttributeError):
            return None

    def reset(self):
        """
        Resets the collection by clearing data and observers.
        """
        self._data = defaultdict(lambda: None)
        self._observers = set(self)
        self.pos_dict = defaultdict(list)
