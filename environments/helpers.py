from collections import defaultdict
from enum import Enum, auto
from typing import Tuple, Union

import numpy as np
from pathlib import Path

LEVELS_DIR = 'levels'

TO_BE_AVERAGED = ['dirt_amount', 'dirty_tiles']
IGNORED_DF_COLUMNS = ['Episode', 'Run', 'train_step', 'step', 'index', 'dirt_amount',
                      'dirty_tile_count', 'terminal_observation', 'episode']


# Constants
class Constants(Enum):
    WALL            = '#'
    DOOR            = 'D'
    DANGER_ZONE     = 'x'
    LEVEL           = 'level'
    AGENT           = 'Agent'
    FREE_CELL       = 0
    OCCUPIED_CELL   = 1
    NO_POS          = (-9999, -9999)

    DOORS           = 'doors'
    CLOSED_DOOR     = 1
    OPEN_DOOR       = -1

    ACTION          = auto()
    COLLISIONS      = auto()
    VALID           = True
    NOT_VALID       = False

    # Dirt Env
    DIRT            = 'dirt'

    # Item Env
    ITEM            = 'item'
    INVENTORY       = 'inventory'

    def __bool__(self):
        return bool(self.value)


class ManhattanMoves(Enum):
    NORTH = 'north'
    EAST = 'east'
    SOUTH = 'south'
    WEST = 'west'


class DiagonalMoves(Enum):
    NORTHEAST = 'north_east'
    SOUTHEAST = 'south_east'
    SOUTHWEST = 'south_west'
    NORTHWEST = 'north_west'


class EnvActions(Enum):
    NOOP        = 'no_op'
    USE_DOOR    = 'use_door'
    CLEAN_UP    = 'clean_up'
    ITEM_ACTION = 'item_action'


d = DiagonalMoves
m = ManhattanMoves
c = Constants

ACTIONMAP = defaultdict(lambda: (0, 0), {m.NORTH.name: (-1, 0), d.NORTHEAST.name: (-1, +1),
                                         m.EAST.name: (0, 1),   d.SOUTHEAST.name: (1, 1),
                                         m.SOUTH.name: (1, 0),  d.SOUTHWEST.name: (+1, -1),
                                         m.WEST.name: (0, -1),  d.NORTHWEST.name: (-1, -1)
                                         }
                        )


# Utility functions
def parse_level(path):
    with path.open('r') as lvl:
        level = list(map(lambda x: list(x.strip()), lvl.readlines()))
    if len(set([len(line) for line in level])) > 1:
        raise AssertionError('Every row of the level string must be of equal length.')
    return level


def one_hot_level(level, wall_char: Union[c, str] = c.WALL):
    grid = np.array(level)
    binary_grid = np.zeros(grid.shape, dtype=np.int8)
    if wall_char in c:
        binary_grid[grid == wall_char.value] = c.OCCUPIED_CELL.value
    else:
        binary_grid[grid == wall_char] = c.OCCUPIED_CELL.value
    return binary_grid


def check_position(slice_to_check_against: np.ndarray, position_to_check: Tuple[int, int]):
    x_pos, y_pos = position_to_check

    # Check if agent colides with grid boundrys
    valid = not (
            x_pos < 0 or y_pos < 0
            or x_pos >= slice_to_check_against.shape[0]
            or y_pos >= slice_to_check_against.shape[0]
    )

    # Check for collision with level walls
    valid = valid and not slice_to_check_against[x_pos, y_pos]
    return c.VALID if valid else c.NOT_VALID


def asset_str(agent):
    # What does this abonimation do?
    # if any([x is None for x in [self._slices[j] for j in agent.collisions]]):
    #     print('error')
    col_names = [x.name for x in agent.temp_collisions]
    if c.AGENT.value in col_names:
        return 'agent_collision', 'blank'
    elif not agent.temp_valid or c.LEVEL.name in col_names or c.AGENT.name in col_names:
        return c.AGENT.value, 'invalid'
    elif agent.temp_valid:
        return c.AGENT.value, 'valid'
    else:
        return c.AGENT.value, 'idle'


if __name__ == '__main__':
    parsed_level = parse_level(Path(__file__).parent / 'factory' / 'levels' / 'simple.txt')
    y = one_hot_level(parsed_level)
    print(np.argwhere(y == 0))
