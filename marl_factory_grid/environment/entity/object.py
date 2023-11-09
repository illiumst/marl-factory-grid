from collections import defaultdict
from typing import Union

from marl_factory_grid.environment import constants as c
import marl_factory_grid.utils.helpers as h


class _Object:
    """Generell Objects for Organisation and Maintanance such as Actions etc..."""

    _u_idx = defaultdict(lambda: 0)

    def __bool__(self):
        return True

    @property
    def var_can_be_bound(self):
        try:
            return self._collection.var_can_be_bound or False
        except AttributeError:
            return False

    @property
    def observers(self):
        return self._observers

    @property
    def name(self):
        return f'{self.__class__.__name__}[{self.identifier}]'

    @property
    def identifier(self):
        if self._str_ident is not None:
            return self._str_ident
        else:
            return self.u_int

    def reset_uid(self):
        self._u_idx = defaultdict(lambda: 0)
        return True

    def __init__(self, str_ident: Union[str, None] = None, **kwargs):
        self._bound_entity = None
        self._observers = []
        self._str_ident = str_ident
        self.u_int = self._identify_and_count_up()
        self._collection = None

        if kwargs:
            print(f'Following kwargs were passed, but ignored: {kwargs}')

    def __repr__(self):
            name = self.name
            if self.bound_entity:
                name = h.add_bound_name(name, self.bound_entity)
            try:
                if self.var_has_position:
                    name = h.add_pos_name(name, self)
            except (AttributeError):
                pass
            return name

    def __eq__(self, other) -> bool:
        return other == self.identifier

    def __hash__(self):
        return hash(self.identifier)

    def _identify_and_count_up(self):
        idx = _Object._u_idx[self.__class__.__name__]
        _Object._u_idx[self.__class__.__name__] += 1
        return idx

    def set_collection(self, collection):
        self._collection = collection

    def add_observer(self, observer):
        self.observers.append(observer)
        observer.notify_add_entity(self)

    def del_observer(self, observer):
        self.observers.remove(observer)

    def summarize_state(self):
        return dict()

    def bind_to(self, entity):
        # noinspection PyAttributeOutsideInit
        self._bound_entity = entity
        return c.VALID

    def belongs_to_entity(self, entity):
        return self._bound_entity == entity

    @property
    def bound_entity(self):
        return self._bound_entity

    def unbind(self):
        self._bound_entity = None


# class EnvObject(_Object):
#     """Objects that hold Information that are observable, but have no position on the environment grid. Inventories etc..."""
#
    # _u_idx = defaultdict(lambda: 0)
#
#     @property
#     def obs_tag(self):
#         try:
#             return self._collection.name or self.name
#         except AttributeError:
#             return self.name
#
#     @property
#     def var_is_blocking_light(self):
#         try:
#             return self._collection.var_is_blocking_light or False
#         except AttributeError:
#             return False
#
#     @property
#     def var_can_be_bound(self):
#         try:
#             return self._collection.var_can_be_bound or False
#         except AttributeError:
#             return False
#
#     @property
#     def var_can_move(self):
#         try:
#             return self._collection.var_can_move or False
#         except AttributeError:
#             return False
#
#     @property
#     def var_is_blocking_pos(self):
#         try:
#             return self._collection.var_is_blocking_pos or False
#         except AttributeError:
#             return False
#
#     @property
#     def var_has_position(self):
#         try:
#             return self._collection.var_has_position or False
#         except AttributeError:
#             return False
#
# @property
# def var_can_collide(self):
#     try:
#         return self._collection.var_can_collide or False
#     except AttributeError:
#         return False
#
#
# @property
# def encoding(self):
#     return c.VALUE_OCCUPIED_CELL
#
#
# def __init__(self, **kwargs):
#     self._bound_entity = None
#     super(EnvObject, self).__init__(**kwargs)
#
#
# def change_parent_collection(self, other_collection):
#     other_collection.add_item(self)
#     self._collection.delete_env_object(self)
#     self._collection = other_collection
#     return self._collection == other_collection
#
#
# def summarize_state(self):
#     return dict(name=str(self.name))
