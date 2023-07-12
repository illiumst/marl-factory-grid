from typing import List

import numpy as np

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.object import EnvObject
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.utils import helpers as h


class Floor(EnvObject):

    @property
    def var_has_position(self):
        return True

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_move(self):
        return False

    @property
    def var_is_blocking_pos(self):
        return False

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def neighboring_floor(self):
        if self._neighboring_floor:
            pass
        else:
            self._neighboring_floor = [x for x in [self._collection.by_pos(np.add(self.pos, pos))
                                                   for pos in h.POS_MASK.reshape(-1, 2)
                                                   if not np.all(pos == [0, 0])]
                                       if x]
        return self._neighboring_floor

    @property
    def encoding(self):
        return c.VALUE_OCCUPIED_CELL

    @property
    def guests_that_can_collide(self):
        return [x for x in self.guests if x.var_can_collide]

    @property
    def guests(self):
        return self._guests.values()

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def is_blocked(self):
        return any([x.var_is_blocking_pos for x in self.guests])

    def __init__(self, pos, **kwargs):
        super(Floor, self).__init__(**kwargs)
        self._guests = dict()
        self.pos = tuple(pos)
        self._neighboring_floor: List[Floor] = list()
        self._blocked_by = None

    def __len__(self):
        return len(self._guests)

    def is_empty(self):
        return not len(self._guests)

    def is_occupied(self):
        return bool(len(self._guests))

    def enter(self, guest):
        if (guest.name not in self._guests and not self.is_blocked) and not (guest.var_is_blocking_pos and self.is_occupied()):
            self._guests.update({guest.name: guest})
            return c.VALID
        else:
            return c.NOT_VALID

    def leave(self, guest):
        try:
            del self._guests[guest.name]
        except (ValueError, KeyError):
            return c.NOT_VALID
        return c.VALID

    def __repr__(self):
        return f'{self.name}(@{self.pos})'

    def summarize_state(self, **_):
        return dict(name=self.name, x=int(self.x), y=int(self.y))

    def render(self):
        return None


class Wall(Floor):

    @property
    def var_can_collide(self):
        return True

    @property
    def encoding(self):
        return c.VALUE_OCCUPIED_CELL

    def render(self):
        return RenderEntity(c.WALL, self.pos)

    @property
    def var_is_blocking_pos(self):
        return True

    @property
    def var_is_blocking_light(self):
        return True
