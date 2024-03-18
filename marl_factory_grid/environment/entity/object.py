from collections import defaultdict
from typing import Union

from marl_factory_grid.environment import constants as c
import marl_factory_grid.utils.helpers as h


class Object:

    _u_idx = defaultdict(lambda: 0)

    @property
    def bound_entity(self):
        """
        Returns the entity to which this object is bound.

        :return: The bound entity.
        """
        return self._bound_entity

    @property
    def var_can_be_bound(self) -> bool:
        """
        Indicates if it is possible to bind this object to another Entity or Object.

        :return: Whether this object can be bound.
        """
        try:
            return self._collection.var_can_be_bound or False
        except AttributeError:
            return False

    @property
    def observers(self) -> set:
        """
        Returns the set of observers for this object.

        :return: Set of observers.
        """
        return self._observers

    @property
    def name(self):
        """
        Returns a string representation of the object's name.

        :return: The name of the object.
        """
        return f'{self.__class__.__name__}[{self.identifier}]'

    @property
    def identifier(self):
        """
        Returns the unique identifier of the object.

        :return: The unique identifier.
        """
        if self._str_ident is not None:
            return self._str_ident
        else:
            return self.u_int

    def reset_uid(self):
        """
        Resets the unique identifier counter for this class.

        :return: True if the reset was successful.
        """
        self._u_idx = defaultdict(lambda: 0)
        return True

    def __init__(self, str_ident: Union[str, None] = None, **kwargs):
        """
        General Objects for Organisation and Maintenance such as Actions, etc.

        :param str_ident: A string identifier for the object.
        :return: None
        """
        self._status = None
        self._bound_entity = None
        self._observers = set()
        self._str_ident = str_ident
        self.u_int = self._identify_and_count_up()
        self._collection = None

        if kwargs:
            print(f'Following kwargs were passed, but ignored: {kwargs}')

    def __bool__(self) -> bool:
        return True

    def __repr__(self):
        name = self.name
        if self.bound_entity:
            name = h.add_bound_name(name, self.bound_entity)
        try:
            if self.var_has_position:
                name = h.add_pos_name(name, self)
        except AttributeError:
            pass
        return name

    def __eq__(self, other) -> bool:
        return other == self.identifier

    def __hash__(self):
        return hash(self.identifier)

    def _identify_and_count_up(self) -> int:
        """Internal Usage"""
        idx = Object._u_idx[self.__class__.__name__]
        Object._u_idx[self.__class__.__name__] += 1
        return idx

    def set_collection(self, collection):
        """Internal Usage"""
        self._collection = collection
        return self

    def add_observer(self, observer):
        """Internal Usage"""
        self.observers.add(observer)
        observer.notify_add_entity(self)
        return self

    def del_observer(self, observer):
        """Internal Usage"""
        self.observers.remove(observer)
        return self

    def summarize_state(self):
        return dict()

    def clear_temp_state(self):
        """Internal Usage"""
        self._status = None
        return self

    def bind_to(self, entity):
        """
        Binds the object to a specified entity.

        :param entity: The entity to bind to.
        :return: The validity of the binding.
        """
        self._bound_entity = entity
        return c.VALID

    def belongs_to_entity(self, entity):
        """
        Checks if the object belongs to a specified entity.

        :param entity: The entity to check against.
        :return: True if the object belongs to the entity, False otherwise.
        """
        return self._bound_entity == entity

    def unbind(self):
        """
        Unbinds the object from its current entity.

        :return: The entity that the object was previously bound to.
        """
        previously_bound = self._bound_entity
        self._bound_entity = None
        return previously_bound
