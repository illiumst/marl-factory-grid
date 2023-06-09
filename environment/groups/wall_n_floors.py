import random
from typing import List

import numpy as np

from environment import constants as c
from environment.groups.env_objects import EnvObjects
from environment.groups.mixins import PositionMixin
from environment.entity.wall_floor import Wall, Floor


class Walls(PositionMixin, EnvObjects):
    _entity = Wall
    symbol = c.SYMBOL_WALL

    def __init__(self, *args, **kwargs):
        super(Walls, self).__init__(*args, **kwargs)
        self._value = c.VALUE_OCCUPIED_CELL

    @classmethod
    def from_coordinates(cls, argwhere_coordinates, *args, **kwargs):
        tiles = cls(*args, **kwargs)
        # noinspection PyTypeChecker
        tiles.add_items([cls._entity(pos) for pos in argwhere_coordinates])
        return tiles

    @classmethod
    def from_tiles(cls, tiles, *args, **kwargs):
        raise RuntimeError()


class Floors(Walls):
    _entity = Floor
    symbol = c.SYMBOL_FLOOR
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super(Floors, self).__init__(*args, **kwargs)
        self._value = c.VALUE_FREE_CELL

    @property
    def occupied_tiles(self):
        tiles = [tile for tile in self if tile.is_occupied()]
        random.shuffle(tiles)
        return tiles

    @property
    def empty_tiles(self) -> List[Floor]:
        tiles = [tile for tile in self if tile.is_empty()]
        random.shuffle(tiles)
        return tiles

    @classmethod
    def from_tiles(cls, tiles, *args, **kwargs):
        raise RuntimeError()
