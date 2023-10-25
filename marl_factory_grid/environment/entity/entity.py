import abc

import numpy as np

from .. import constants as c
from .object import EnvObject
from ...utils.utility_classes import RenderEntity
from ...utils.results import ActionResult


class Entity(EnvObject, abc.ABC):
    """Full Env Entity that lives on the environment Grid. Doors, Items, DirtPile etc..."""

    @property
    def state(self):
        return self._status or ActionResult(entity=self, identifier=c.NOOP, validity=c.VALID, reward=0)

    @property
    def var_has_position(self):
        return self.pos != c.VALUE_NO_POS

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def pos(self):
        return self._pos

    @property
    def last_pos(self):
        try:
            return self._last_pos
        except AttributeError:
            # noinspection PyAttributeOutsideInit
            self._last_pos = c.VALUE_NO_POS
            return self._last_pos

    @property
    def direction_of_view(self):
        if self._last_pos != c.VALUE_NO_POS:
            return 0, 0
        else:
            return np.subtract(self._last_pos, self.pos)

    def move(self, next_pos, state):
        next_pos = next_pos
        curr_pos = self._pos
        if not_same_pos := curr_pos != next_pos:
            if valid := state.check_move_validity(self, next_pos):
                for observer in self.observers:
                    observer.notify_del_entity(self)
                self._view_directory = curr_pos[0]-next_pos[0], curr_pos[1]-next_pos[1]
                self._pos = next_pos
                for observer in self.observers:
                    observer.notify_add_entity(self)
            return valid
        return not_same_pos

    def __init__(self, pos, bind_to=None, **kwargs):
        super().__init__(**kwargs)
        self._status = None
        self._pos = pos
        self._last_pos = pos
        if bind_to:
            try:
                self.bind_to(bind_to)
            except AttributeError:
                print(f'Objects of class "{self.__class__.__name__}" can not be bound to other entities.')
                exit()

    def summarize_state(self) -> dict:  # tile=str(self.tile.name)
        return dict(name=str(self.name), x=int(self.x), y=int(self.y), can_collide=bool(self.var_can_collide))

    @abc.abstractmethod
    def render(self):
        return RenderEntity(self.__class__.__name__.lower(), self.pos)

    def __repr__(self):
        return super(Entity, self).__repr__() + f'(@{self.pos})'
