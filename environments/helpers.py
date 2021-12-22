import itertools
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Tuple, Union, Dict, List

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from stable_baselines3 import PPO, DQN, A2C

MODEL_MAP = dict(PPO=PPO, DQN=DQN, A2C=A2C)

LEVELS_DIR = 'levels'
STEPS_START = 1

TO_BE_AVERAGED = ['dirt_amount', 'dirty_tiles']
IGNORED_DF_COLUMNS = ['Episode', 'Run', 'train_step', 'step', 'index', 'dirt_amount',
                      'dirty_tile_count', 'terminal_observation', 'episode']


# Constants
class Constants(Enum):
    WALL                = '#'
    WALLS               = 'Walls'
    FLOOR               = 'Floor'
    DOOR                = 'D'
    DANGER_ZONE         = 'x'
    LEVEL               = 'Level'
    AGENT               = 'Agent'
    AGENT_PLACEHOLDER   = 'AGENT_PLACEHOLDER'
    GLOBAL_POSITION     = 'GLOBAL_POSITION'
    FREE_CELL           = 0
    OCCUPIED_CELL       = 1
    SHADOWED_CELL       = -1
    NO_POS              = (-9999, -9999)

    DOORS               = 'Doors'
    CLOSED_DOOR         = 'closed'
    OPEN_DOOR           = 'open'

    ACTION              = 'action'
    COLLISIONS          = 'collision'
    VALID               = 'valid'
    NOT_VALID           = 'not_valid'

    # Dirt Env
    DIRT                = 'Dirt'

    # Item Env
    ITEM                = 'Item'
    INVENTORY           = 'Inventory'
    DROP_OFF            = 'Drop_Off'

    # Battery Env
    CHARGE_POD          = 'Charge_Pod'
    BATTERIES           = 'BATTERIES'

    # Destination Env
    DESTINATION         = 'Destination'
    REACHEDDESTINATION  = 'ReachedDestination'

    def __bool__(self):
        if 'not_' in self.value:
            return False
        else:
            return bool(self.value)


class MovingAction(Enum):
    NORTH = 'north'
    EAST = 'east'
    SOUTH = 'south'
    WEST = 'west'
    NORTHEAST = 'north_east'
    SOUTHEAST = 'south_east'
    SOUTHWEST = 'south_west'
    NORTHWEST = 'north_west'

    @classmethod
    def is_member(cls, other):
        return any([other == direction for direction in cls])

    @classmethod
    def square(cls):
        return [cls.NORTH, cls.EAST, cls.SOUTH, cls.WEST]

    @classmethod
    def diagonal(cls):
        return [cls.NORTHEAST, cls.SOUTHEAST, cls.SOUTHWEST, cls.NORTHWEST]


class EnvActions(Enum):
    NOOP            = 'no_op'
    USE_DOOR        = 'use_door'
    CLEAN_UP        = 'clean_up'
    ITEM_ACTION     = 'item_action'
    CHARGE          = 'charge'
    WAIT_ON_DEST    = 'wait'


m = MovingAction
c = Constants

ACTIONMAP = defaultdict(lambda: (0, 0), {m.NORTH: (-1, 0), m.NORTHEAST: (-1, +1),
                                         m.EAST: (0, 1),   m.SOUTHEAST: (1, 1),
                                         m.SOUTH: (1, 0),  m.SOUTHWEST: (+1, -1),
                                         m.WEST: (0, -1),  m.NORTHWEST: (-1, -1)
                                         }
                        )


class ObservationTranslator:

    def __init__(self, obs_shape_2d: (int, int), this_named_observation_space: Dict[str, dict],
                 *per_agent_named_obs_space: Dict[str, dict],
                 placeholder_fill_value: Union[int, str] = 'N'):
        assert len(obs_shape_2d) == 2
        self.obs_shape = obs_shape_2d
        if isinstance(placeholder_fill_value, str):
            if placeholder_fill_value.lower() in ['normal', 'n']:
                self.random_fill = lambda: np.random.normal(size=self.obs_shape)
            elif placeholder_fill_value.lower() in ['uniform', 'u']:
                self.random_fill = lambda: np.random.uniform(size=self.obs_shape)
            else:
                raise ValueError('Please chooe between "uniform" or "normal"')
        else:
            self.random_fill = None

        self._this_named_obs_space = this_named_observation_space
        self._per_agent_named_obs_space = list(per_agent_named_obs_space)

    def translate_observation(self, agent_idx: int, obs: np.ndarray):
        target_obs_space = self._per_agent_named_obs_space[agent_idx]
        translation = [idx_space_dict['explained_idxs'] for name, idx_space_dict in target_obs_space.items()]
        flat_translation = [x for y in translation for x in y]
        return np.take(obs, flat_translation, axis=1 if obs.ndim == 4 else 0)

    def translate_observations(self, observations: List[ArrayLike]):
        return [self.translate_observation(idx, observation) for idx, observation in enumerate(observations)]

    def __call__(self, observations):
        return self.translate_observations(observations)


class ActionTranslator:

    def __init__(self, target_named_action_space: Dict[str, int], *per_agent_named_action_space: Dict[str, int]):
        self._target_named_action_space = target_named_action_space
        self._per_agent_named_action_space = list(per_agent_named_action_space)
        self._per_agent_idx_actions = [{idx: a for a, idx in x.items()} for x in self._per_agent_named_action_space]

    def translate_action(self, agent_idx: int, action: int):
        named_action = self._per_agent_idx_actions[agent_idx][action]
        translated_action = self._target_named_action_space[named_action]
        return translated_action

    def translate_actions(self, actions: List[int]):
        return [self.translate_action(idx, action) for idx, action in enumerate(actions)]

    def __call__(self, actions):
        return self.translate_actions(actions)


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


def check_position(slice_to_check_against: ArrayLike, position_to_check: Tuple[int, int]):
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
    if any(c.AGENT.value in name for name in col_names):
        return 'agent_collision', 'blank'
    elif not agent.temp_valid or c.LEVEL.name in col_names or c.AGENT.name in col_names:
        return c.AGENT.value, 'invalid'
    elif agent.temp_valid and not MovingAction.is_member(agent.temp_action):
        return c.AGENT.value, 'valid'
    elif agent.temp_valid and MovingAction.is_member(agent.temp_action):
        return c.AGENT.value, 'move'
    else:
        return c.AGENT.value, 'idle'


def points_to_graph(coordiniates_or_tiles, allow_euclidean_connections=True, allow_manhattan_connections=True):
    assert allow_euclidean_connections or allow_manhattan_connections
    if hasattr(coordiniates_or_tiles, 'positions'):
        coordiniates_or_tiles = coordiniates_or_tiles.positions
    possible_connections = itertools.combinations(coordiniates_or_tiles, 2)
    graph = nx.Graph()
    for a, b in possible_connections:
        diff = abs(np.subtract(a, b))
        if not max(diff) > 1:
            if allow_manhattan_connections and allow_euclidean_connections:
                graph.add_edge(a, b)
            elif not allow_manhattan_connections and allow_euclidean_connections and all(diff):
                graph.add_edge(a, b)
            elif allow_manhattan_connections and not allow_euclidean_connections and not all(diff) and any(diff):
                graph.add_edge(a, b)
    return graph


if __name__ == '__main__':
    parsed_level = parse_level(Path(__file__).parent / 'factory' / 'levels' / 'simple.txt')
    y = one_hot_level(parsed_level)
    print(np.argwhere(y == 0))
