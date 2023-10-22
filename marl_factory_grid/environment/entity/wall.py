from typing import List

import numpy as np

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.entity.object import EnvObject
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.utils import helpers as h


class Wall(Entity):

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
