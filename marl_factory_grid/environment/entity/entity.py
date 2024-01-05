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
        Get the current status of the entity. Not to be confused with the Gamestate.
        :return: status
        """
        return self._status or State(entity=self, identifier=c.NOOP, validity=c.VALID)

    @property
    def var_has_position(self):
        """
        Check if the entity has a position.

        :return: True if the entity has a position, False otherwise.
        :rtype: bool
        """
        return self.pos != c.VALUE_NO_POS

    @property
    def var_is_blocking_light(self):
        """
        Check if the entity is blocking light.

        :return: True if the entity is blocking light, False otherwise.
        :rtype: bool
        """
        try:
            return self._collection.var_is_blocking_light or False
        except AttributeError:
            return False

    @property
    def var_can_move(self):
        """
        Check if the entity can move.

        :return: True if the entity can move, False otherwise.
        :rtype: bool
        """
        try:
            return self._collection.var_can_move or False
        except AttributeError:
            return False

    @property
    def var_is_blocking_pos(self):
        """
        Check if the entity is blocking a position when standing on it.

        :return: True if the entity is blocking a position, False otherwise.
        :rtype: bool
        """
        try:
            return self._collection.var_is_blocking_pos or False
        except AttributeError:
            return False

    @property
    def var_can_collide(self):
        """
        Check if the entity can collide.

        :return: True if the entity can collide, False otherwise.
        :rtype: bool
        """
        try:
            return self._collection.var_can_collide or False
        except AttributeError:
            return False

    @property
    def x(self):
        """
        Get the x-coordinate of the entity's position.

        :return: The x-coordinate of the entity's position.
        :rtype: int
        """
        return self.pos[0]

    @property
    def y(self):
        """
        Get the y-coordinate of the entity's position.

        :return: The y-coordinate of the entity's position.
        :rtype: int
        """
        return self.pos[1]

    @property
    def pos(self):
        """
        Get the current position of the entity.

        :return: The current position of the entity.
        :rtype: tuple
        """
        return self._pos

    def set_pos(self, pos) -> bool:
        """
        Set the position of the entity.

        :param pos: The new position.
        :type pos: tuple
        :return: True if setting the position is successful, False otherwise.
        """
        assert isinstance(pos, tuple) and len(pos) == 2
        self._pos = pos
        return c.VALID

    @property
    def last_pos(self):
        """
        Get the last position of the entity.

        :return: The last position of the entity.
        :rtype: tuple
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
        Get the current direction of view of the entity.

        :return: The current direction of view of the entity.
        :rtype: int
        """
        if self._last_pos != c.VALUE_NO_POS:
            return 0, 0
        else:
            return np.subtract(self._last_pos, self.pos)

    def __init__(self, pos, bind_to=None, **kwargs):
        """
        Abstract base class representing entities in the environment grid.

        :param pos: The initial position of the entity.
        :type pos: tuple
        :param bind_to: Entity to which this entity is bound (Default: None)
        :type bind_to: Entity or None
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

    def move(self, next_pos, state):
        """
        Move the entity to a new position.

        :param next_pos: The next position to move the entity to.
        :type next_pos: tuple
        :param state: The current state of the environment.
        :type state: marl_factory_grid.environment.state.Gamestate

        :return: True if the move is valid, False otherwise.
        :rtype: bool
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

    def summarize_state(self) -> dict:
        """
        Summarize the current state of the entity.

        :return: A dictionary containing the name, x-coordinate, y-coordinate, and can_collide property of the entity.
        :rtype: dict
        """
        return dict(name=str(self.name), x=int(self.x), y=int(self.y), can_collide=bool(self.var_can_collide))

    @abc.abstractmethod
    def render(self):
        """
        Abstract method to render the entity.

        :return: A rendering entity representing the entity's appearance in the environment.
        :rtype: marl_factory_grid.utils.utility_classes.RenderEntity
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
        Get the encoded representation of the entity.

        :return: The encoded representation.
        :rtype: int
        """
        return c.VALUE_OCCUPIED_CELL

    def change_parent_collection(self, other_collection):
        """
        Change the parent collection of the entity.

        :param other_collection: The new parent collection.
        :type other_collection: marl_factory_grid.environment.collections.Collection

        :return: True if the change is successful, False otherwise.
        :rtype: bool
        """
        other_collection.add_item(self)
        self._collection.delete_env_object(self)
        self._collection = other_collection
        return self._collection == other_collection

    @property
    def collection(self):
        """
        Get the parent collection of the entity.

        :return: The parent collection.
        :rtype: marl_factory_grid.environment.collections.Collection
        """
        return self._collection
