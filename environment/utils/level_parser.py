from os import PathLike
from pathlib import Path
from typing import Dict

import numpy as np

from environment.groups.global_entities import Entities
from environment.groups.wall_n_floors import Walls, Floors
from environment.utils import helpers as h
from environment import constants as c


class LevelParser(object):

    @property
    def pomdp_d(self):
        return self.pomdp_r * 2 + 1

    def __init__(self, level_file_path: PathLike, entity_parse_dict: Dict[Entities, dict], pomdp_r=0):
        self.pomdp_r = pomdp_r
        self.e_p_dict = entity_parse_dict
        self._parsed_level = h.parse_level(Path(level_file_path))
        level_array = h.one_hot_level(self._parsed_level, c.SYMBOL_WALL)
        self.level_shape = level_array.shape
        self.size = self.pomdp_r**2 if self.pomdp_r else np.prod(self.level_shape)

    def do_init(self):
        entities = Entities()
        # Walls
        level_array = h.one_hot_level(self._parsed_level, c.SYMBOL_WALL)

        walls = Walls.from_coordinates(np.argwhere(level_array == c.VALUE_OCCUPIED_CELL), self.size)
        entities.add_items({c.WALL: walls})

        # Floor
        floor = Floors.from_coordinates(np.argwhere(level_array == c.VALUE_FREE_CELL), self.size)
        entities.add_items({c.FLOOR: floor})

        # All other
        for es_name in self.e_p_dict:
            e_class, e_kwargs = self.e_p_dict[es_name]['class'], self.e_p_dict[es_name]['kwargs']

            if hasattr(e_class, 'symbol'):
                level_array = h.one_hot_level(self._parsed_level, symbol=e_class.symbol)
                if np.any(level_array):
                    e = e_class.from_coordinates(np.argwhere(level_array == c.VALUE_OCCUPIED_CELL).tolist(),
                                                 entities[c.FLOOR], self.size, entity_kwargs=e_kwargs
                                                 )
                else:
                    raise ValueError(f'No {e_class} (Symbol: {e_class.symbol}) could be found!\n'
                                     f'Check your level file!')
            else:
                e = e_class(self.size, **e_kwargs)
            entities.add_items({e.name: e})
        return entities
