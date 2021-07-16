from collections import defaultdict
from enum import Enum, auto
from typing import Tuple, Union

import numpy as np
from pathlib import Path


# Constants
class Constants(Enum):
    WALL = '#'
    DOOR = 'D'
    DANGER_ZONE = 'x'
    LEVEL = 'level'
    AGENT = 'Agent'
    FREE_CELL = 0
    OCCUPIED_CELL = 1

    DOORS = 'doors'
    CLOSED_DOOR = 1
    OPEN_DOOR = -1

    LEVEL_IDX = 0

    ACTION = auto()
    COLLISIONS = auto()
    VALID = True
    NOT_VALID = False

    def __bool__(self):
        return bool(self.value)


LEVELS_DIR = 'levels'

TO_BE_AVERAGED = ['dirt_amount', 'dirty_tiles']
IGNORED_DF_COLUMNS = ['Episode', 'Run', 'train_step', 'step', 'index', 'dirt_amount',
                      'dirty_tile_count', 'terminal_observation', 'episode']

MANHATTAN_MOVES = ['north', 'east', 'south', 'west']
DIAGONAL_MOVES = ['north_east', 'south_east', 'south_west', 'north_west']

NO_POS = (-9999, -9999)

ACTIONMAP = defaultdict(lambda: (0, 0), dict(north=(-1, 0), east=(0, 1),
                                             south=(1, 0), west=(0, -1),
                                             north_east=(-1, +1), south_east=(1, 1),
                                             south_west=(+1, -1), north_west=(-1, -1)
                                             )
                        )

HORIZONTAL_DOOR_MAP = np.asarray([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
VERTICAL_DOOR_MAP = np.asarray([[0, 1, 0], [0, 0, 0], [0, 1, 0]])

HORIZONTAL_DOOR_ZONE_1 = np.asarray([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
HORIZONTAL_DOOR_ZONE_2 = np.asarray([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
VERTICAL_DOOR_ZONE_1 = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
VERTICAL_DOOR_ZONE_2 = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 1]])




# Utility functions
def parse_level(path):
    with path.open('r') as lvl:
        level = list(map(lambda x: list(x.strip()), lvl.readlines()))
    if len(set([len(line) for line in level])) > 1:
        raise AssertionError('Every row of the level string must be of equal length.')
    return level


def one_hot_level(level, wall_char: Union[Constants, str] = Constants.WALL):
    grid = np.array(level)
    binary_grid = np.zeros(grid.shape, dtype=np.int8)
    if wall_char in Constants:
        binary_grid[grid == wall_char.value] = Constants.OCCUPIED_CELL.value
    else:
        binary_grid[grid == wall_char] = Constants.OCCUPIED_CELL.value
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
    return Constants.VALID if valid else Constants.NOT_VALID


if __name__ == '__main__':
    parsed_level = parse_level(Path(__file__).parent / 'factory' / 'levels' / 'simple.txt')
    y = one_hot_level(parsed_level)
    print(np.argwhere(y == 0))
