import abc

import numpy as np

from .object import Object
from .. import constants as c
from ...utils.results import State
from ...utils.utility_classes import RenderEntity


class Entity(Object, abc.ABC):

    @property
    def state(self):
        """
        TODO


        :return:
        """
        return self._status or State(entity=self, identifier=c.NOOP, validity=c.VALID)

    @property
    def var_has_position(self):
        """
        TODO


        :return:
        """
        return self.pos != c.VALUE_NO_POS

    @property
    def var_is_blocking_light(self):
        """
        TODO


        :return:
        """
        try:
            return self._collection.var_is_blocking_light or False
        except AttributeError:
            return False

    @property
    def var_can_move(self):
        """
        TODO


        :return:
        """
        try:
            return self._collection.var_can_move or False
        except AttributeError:
            return False

    @property
    def var_is_blocking_pos(self):
        """
        TODO


        :return:
        """
        try:
            return self._collection.var_is_blocking_pos or False
        except AttributeError:
            return False

    @property
    def var_can_collide(self):
        """
        TODO


        :return:
        """
        try:
            return self._collection.var_can_collide or False
        except AttributeError:
            return False

    @property
    def x(self):
        """
        TODO


        :return:
        """
        return self.pos[0]

    @property
    def y(self):
        """
        TODO


        :return:
        """
        return self.pos[1]

    @property
    def pos(self):
        """
        TODO


        :return:
        """
        return self._pos

    def set_pos(self, pos) -> bool:
        """
        TODO


        :return:
        """
        assert isinstance(pos, tuple) and len(pos) == 2
        self._pos = pos
        return c.VALID

    @property
    def last_pos(self):
        """
        TODO


        :return:
        """
        try:
            return self._last_pos
        except AttributeError:
            # noinspection PyAttributeOutsideInit
            self._last_pos = c.VALUE_NO_POS
            return self._last_pos

    @property
    def direction_of_view(self):
        """
        TODO


        :return:
        """
        if self._last_pos != c.VALUE_NO_POS:
            return 0, 0
        else:
            return np.subtract(self._last_pos, self.pos)

    def move(self, next_pos, state):
        """
        TODO


        :return:
        """
        next_pos = next_pos
        curr_pos = self._pos
        if not_same_pos := curr_pos != next_pos:
            if valid := state.check_move_validity(self, next_pos):
                for observer in self.observers:
                    observer.notify_del_entity(self)
                self._view_directory = curr_pos[0] - next_pos[0], curr_pos[1] - next_pos[1]
                self.set_pos(next_pos)
                for observer in self.observers:
                    observer.notify_add_entity(self)
            return valid
        # Bad naming... Was the same was the same pos, not moving....
        return not_same_pos

    def __init__(self, pos, bind_to=None, **kwargs):
        """
        Full Env Entity that lives on the environment Grid. Doors, Items, DirtPile etc...
        TODO


        :return:
        """
        super().__init__(**kwargs)
        self._view_directory = c.VALUE_NO_POS
        self._status = None
        self._pos = pos
        self._last_pos = pos
        self._collection = None
        if bind_to:
            try:
                self.bind_to(bind_to)
            except AttributeError:
                print(f'Objects of class "{self.__class__.__name__}" can not be bound to other entities.')
                exit()

    def summarize_state(self) -> dict:
        """
        TODO


        :return:
        """
        return dict(name=str(self.name), x=int(self.x), y=int(self.y), can_collide=bool(self.var_can_collide))

    @abc.abstractmethod
    def render(self):
        """
        TODO


        :return:
        """
        return RenderEntity(self.__class__.__name__.lower(), self.pos)

    @property
    def obs_tag(self):
        """Internal Usage"""
        try:
            return self._collection.name or self.name
        except AttributeError:
            return self.name

    @property
    def encoding(self):
        """
        TODO


        :return:
        """
        return c.VALUE_OCCUPIED_CELL

    def change_parent_collection(self, other_collection):
        """
        TODO


        :return:
        """
        other_collection.add_item(self)
        self._collection.delete_env_object(self)
        self._collection = other_collection
        return self._collection == other_collection

    @property
    def collection(self):
        """
        TODO


        :return:
        """
        return self._collection
