import random
from typing import List, Tuple

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin
from marl_factory_grid.environment.entity.wall_floor import Wall, Floor


class Walls(PositionMixin, EnvObjects):
    _entity = Wall
    symbol = c.SYMBOL_WALL

    def __init__(self, *args, **kwargs):
        super(Walls, self).__init__(*args, **kwargs)
        self._value = c.VALUE_OCCUPIED_CELL

    #ToDo: Do we need this? Move to spawn methode?
    # @classmethod
    # def from_coordinates(cls, argwhere_coordinates, *args, **kwargs):
    #     tiles = cls(*args, **kwargs)
    #     # noinspection PyTypeChecker
    #     tiles.add_items([cls._entity(pos) for pos in argwhere_coordinates])
    #     return tiles

    def by_pos(self, pos: (int, int)):
        try:
            return super().by_pos(pos)[0]
        except IndexError:
            return None


class Floors(Walls):
    _entity = Floor
    symbol = c.SYMBOL_FLOOR
    var_is_blocking_light: bool = False
    var_can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super(Floors, self).__init__(*args, **kwargs)
        self._value = c.VALUE_FREE_CELL

