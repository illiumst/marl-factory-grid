import itertools
from collections import defaultdict
from typing import Tuple, Union, Dict, List, NamedTuple

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from stable_baselines3 import PPO, DQN, A2C


"""
This file is used for:
    1. string based definition
        Use a class like `Constants`, to define attributes, which then reveal strings.
        These can be used for naming convention along the environments as well as keys for mappings such as dicts etc.
        When defining new envs, use class inheritance. 
    
    2. utility function definition
        There are static utility functions which are not bound to a specific environment.
        In this file they are defined to be used across the entire package.
"""


MODEL_MAP = dict(PPO=PPO, DQN=DQN, A2C=A2C)      # For use in studies and experiments


LEVELS_DIR = 'levels'                            # for use in studies and experiments
STEPS_START = 1                                  # Define where to the stepcount; which is the first step

# Not used anymore? Clean!
# TO_BE_AVERAGED = ['dirt_amount', 'dirty_tiles']
IGNORED_DF_COLUMNS = ['Episode', 'Run',          # For plotting, which values are ignored when loading monitor files
                      'train_step', 'step', 'index', 'dirt_amount', 'dirty_tile_count', 'terminal_observation',
                      'episode']


class Constants:

    """
        String based mapping. Use these to handle keys or define values, which can be then be used globaly.
        Please use class inheritance when defining new environments.
    """

    WALL                = '#'                   # Wall tile identifier for resolving the string based map files.
    DOOR                = 'D'                   # Door identifier for resolving the string based map files.
    DANGER_ZONE         = 'x'                   # Dange Zone tile identifier for resolving the string based map files.

    WALLS               = 'Walls'               # Identifier of Wall-objects and sets (collections).
    FLOOR               = 'Floor'               # Identifier of Floor-objects and sets (collections).
    DOORS               = 'Doors'               # Identifier of Door-objects and sets (collections).
    LEVEL               = 'Level'               # Identifier of Level-objects and sets (collections).
    AGENT               = 'Agent'               # Identifier of Agent-objects and sets (collections).
    AGENT_PLACEHOLDER   = 'AGENT_PLACEHOLDER'   # Identifier of Placeholder-objects and sets (collections).
    GLOBAL_POSITION     = 'GLOBAL_POSITION'     # Identifier of the global position slice

    FREE_CELL           = 0                     # Free-Cell value used in observation
    OCCUPIED_CELL       = 1                     # Occupied-Cell value used in observation
    SHADOWED_CELL       = -1                    # Shadowed-Cell value used in observation
    ACCESS_DOOR_CELL    = 1/3                   # Access-door-Cell value used in observation
    OPEN_DOOR_CELL      = 2/3                   # Open-door-Cell value used in observation
    CLOSED_DOOR_CELL    = 3/3                   # Closed-door-Cell value used in observation

    NO_POS              = (-9999, -9999)        # Invalid Position value used in the environment (something is off-grid)

    CLOSED_DOOR         = 'closed'              # Identifier to compare door-is-closed state
    OPEN_DOOR           = 'open'                # Identifier to compare door-is-open state
    # ACCESS_DOOR         = 'access'            # Identifier to compare access positions

    ACTION              = 'action'              # Identifier of Action-objects and sets (collections).
    COLLISION           = 'collision'           # Identifier to use in the context of collitions.
    VALID               = True                  # Identifier to rename boolean values in the context of actions.
    NOT_VALID           = False                 # Identifier to rename boolean values in the context of actions.


class EnvActions:
    """
        String based mapping. Use these to identifiy actions, can be used globaly.
        Please use class inheritance when defining new environments with new actions.
    """
    # Movements
    NORTH           = 'north'
    EAST            = 'east'
    SOUTH           = 'south'
    WEST            = 'west'
    NORTHEAST       = 'north_east'
    SOUTHEAST       = 'south_east'
    SOUTHWEST       = 'south_west'
    NORTHWEST       = 'north_west'

    # Other
    # MOVE            = 'move'
    NOOP            = 'no_op'
    USE_DOOR        = 'use_door'

    _ACTIONMAP = defaultdict(lambda: (0, 0),
                            {NORTH: (-1, 0),    NORTHEAST: (-1, 1),
                             EAST:  (0, 1),     SOUTHEAST: (1, 1),
                             SOUTH: (1, 0),     SOUTHWEST: (1, -1),
                             WEST:  (0, -1),    NORTHWEST: (-1, -1)
                             }
                            )

    @classmethod
    def is_move(cls, action):
        """
        Classmethod; checks if given action is a movement action or not. Depending on the env. configuration,
        Movement actions are either `manhattan` (square) style movements (up,down, left, right) and/or diagonal.

        :param action:  Action to be checked
        :type action:   str
        :return:        Whether the given action is a movement action.
        :rtype:         bool
        """
        return any([action == direction for direction in cls.movement_actions()])

    @classmethod
    def square_move(cls):
        """
        Classmethod; return a list of movement actions that are considered square or `manhattan` style movements.

        :return: A list of movement actions.
        :rtype: list(str)
        """
        return [cls.NORTH, cls.EAST, cls.SOUTH, cls.WEST]

    @classmethod
    def diagonal_move(cls):
        """
        Classmethod; return a list of movement actions that are considered diagonal movements.

        :return: A list of movement actions.
        :rtype: list(str)
        """
        return [cls.NORTHEAST, cls.SOUTHEAST, cls.SOUTHWEST, cls.NORTHWEST]

    @classmethod
    def movement_actions(cls):
        """
        Classmethod; return a list of all available movement actions.
        Please note, that this is indipendent from the env. properties

        :return: A list of movement actions.
        :rtype: list(str)
        """
        return list(itertools.chain(cls.square_move(), cls.diagonal_move()))

    @classmethod
    def resolve_movement_action_to_coords(cls, action):
        """
        Classmethod; resolve movement actions. Given a movement action, return the delta in coordinates it stands for.
        How does the current entity coordinate change if it performs the given action?
        Please note, this is indipendent from the env. properties

        :return: Delta coorinates.
        :rtype: tuple(int, int)
        """
        return cls._ACTIONMAP[action]


class RewardsBase(NamedTuple):
    """
        Value based mapping. Use these to define reward values for specific conditions (i.e. the action
        in a given context), can be used globaly.
        Please use class inheritance when defining new environments with new rewards.
    """
    MOVEMENTS_VALID: float = -0.001
    MOVEMENTS_FAIL: float  = -0.05
    NOOP: float            = -0.01
    USE_DOOR_VALID: float  = -0.00
    USE_DOOR_FAIL: float   = -0.01
    COLLISION: float       = -0.5


class ObservationTranslator:

    def __init__(self, this_named_observation_space: Dict[str, dict],
                 *per_agent_named_obs_spaces: Dict[str, dict],
                 placeholder_fill_value: Union[int, str, None] = None):
        """
        This is a helper class, which converts agents observations from joined environments.
        For example, agents trained in different environments may expect different observations.
        This class translates from larger observations spaces to smaller.
        A string identifier based approach is used.
        Currently, it is not possible to mix different obs shapes.


        :param this_named_observation_space: `Named observation space` of the joined environment.
        :type  this_named_observation_space: Dict[str, dict]

        :param per_agent_named_obs_spaces: `Named observation space` one for each agent. Overloaded.
        type  per_agent_named_obs_spaces: Dict[str, dict]

        :param placeholder_fill_value: Currently not fully implemented!!!
        :type  placeholder_fill_value: Union[int, str] = 'N')
        """

        if isinstance(placeholder_fill_value, str):
            if placeholder_fill_value.lower() in ['normal', 'n']:
                self.random_fill = np.random.normal
            elif placeholder_fill_value.lower() in ['uniform', 'u']:
                self.random_fill = np.random.uniform
            else:
                raise ValueError('Please chooe between "uniform" or "normal" ("u", "n").')
        elif isinstance(placeholder_fill_value, int):
            raise NotImplementedError('"Future Work."')
        else:
            self.random_fill = None

        self._this_named_obs_space = this_named_observation_space
        self._per_agent_named_obs_space = list(per_agent_named_obs_spaces)

    def translate_observation(self, agent_idx: int, obs: np.ndarray):
        target_obs_space = self._per_agent_named_obs_space[agent_idx]
        translation = dict()
        for name, idxs in target_obs_space.items():
            if name in self._this_named_obs_space:
                for target_idx, this_idx in zip(idxs, self._this_named_obs_space[name]):
                    taken_slice = np.take(obs, [this_idx], axis=1 if obs.ndim == 4 else 0)
                    translation[target_idx] = taken_slice
            elif random_fill := self.random_fill:
                for target_idx in idxs:
                    translation[target_idx] = random_fill(size=obs.shape[:-3] + (1,) + obs.shape[-2:])
            else:
                for target_idx in idxs:
                    translation[target_idx] = np.zeros(shape=(obs.shape[:-3] + (1,) + obs.shape[-2:]))

        translation = dict(sorted(translation.items()))
        return np.concatenate(list(translation.values()), axis=-3)

    def translate_observations(self, observations: List[ArrayLike]):
        return [self.translate_observation(idx, observation) for idx, observation in enumerate(observations)]

    def __call__(self, observations):
        return self.translate_observations(observations)


class ActionTranslator:

    def __init__(self, target_named_action_space: Dict[str, int], *per_agent_named_action_space: Dict[str, int]):
        """
        This is a helper class, which converts agents action spaces to a joined environments action space.
        For example, agents trained in different environments may have different action spaces.
        This class translates from smaller individual agent action spaces to larger joined spaces.
        A string identifier based approach is used.

        :param target_named_action_space:  Joined `Named action space` for the current environment.
        :type target_named_action_space: Dict[str, dict]

        :param per_agent_named_action_space: `Named action space` one for each agent. Overloaded.
        :type per_agent_named_action_space: Dict[str, dict]
        """

        self._target_named_action_space = target_named_action_space
        if isinstance(per_agent_named_action_space, (list, tuple)):
            self._per_agent_named_action_space = per_agent_named_action_space
        else:
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
    """
    Given the path to a strin based `level` or `map` representation, this function reads the content.
    Cleans `space`, checks for equal length of each row and returns a list of lists.

    :param path: Path to the `level` or `map` file on harddrive.
    :type path: os.Pathlike

    :return: The read string representation of the `level` or `map`
    :rtype: List[List[str]]
    """
    with path.open('r') as lvl:
        level = list(map(lambda x: list(x.strip()), lvl.readlines()))
    if len(set([len(line) for line in level])) > 1:
        raise AssertionError('Every row of the level string must be of equal length.')
    return level


def one_hot_level(level, wall_char: str = Constants.WALL):
    """
    Given a string based level representation (list of lists, see function `parse_level`), this function creates a
    binary numpy array or `grid`. Grid values that equal `wall_char` become of `Constants.OCCUPIED_CELL` value.
    Can be changed to filter for any symbol.

    :param level: String based level representation (list of lists, see function `parse_level`).
    :param wall_char: List[List[str]]

    :return: Binary numpy array
    :rtype: np.typing._array_like.ArrayLike
    """

    grid = np.array(level)
    binary_grid = np.zeros(grid.shape, dtype=np.int8)
    binary_grid[grid == wall_char] = Constants.OCCUPIED_CELL
    return binary_grid


def check_position(slice_to_check_against: ArrayLike, position_to_check: Tuple[int, int]):
    """
    Given a slice (2-D Arraylike object)

    :param slice_to_check_against: The slice to check for accessability
    :type slice_to_check_against: np.typing._array_like.ArrayLike

    :param position_to_check: Position in slice that should be checked. Can be outside of slice boundarys.
    :type position_to_check: tuple(int, int)

    :return: Whether a position can be moved to.
    :rtype: bool
    """
    x_pos, y_pos = position_to_check

    # Check if agent colides with grid boundrys
    valid = not (
            x_pos < 0 or y_pos < 0
            or x_pos >= slice_to_check_against.shape[0]
            or y_pos >= slice_to_check_against.shape[1]
    )

    # Check for collision with level walls
    valid = valid and not slice_to_check_against[x_pos, y_pos]
    return Constants.VALID if valid else Constants.NOT_VALID


def asset_str(agent):
    """
        FIXME @ romue
    """
    # What does this abonimation do?
    # if any([x is None for x in [cls._slices[j] for j in agent.collisions]]):
    #     print('error')
    if step_result := agent.step_result:
        action = step_result['action_name']
        valid = step_result['action_valid']
        col_names = [x.name for x in step_result['collisions']]
        if any(Constants.AGENT in name for name in col_names):
            return 'agent_collision', 'blank'
        elif not valid or Constants.LEVEL in col_names or Constants.AGENT in col_names:
            return Constants.AGENT, 'invalid'
        elif valid and not EnvActions.is_move(action):
            return Constants.AGENT, 'valid'
        elif valid and EnvActions.is_move(action):
            return Constants.AGENT, 'move'
        else:
            return Constants.AGENT, 'idle'
    else:
        return Constants.AGENT, 'idle'


def points_to_graph(coordiniates_or_tiles, allow_euclidean_connections=True, allow_manhattan_connections=True):
    """
    Given a set of coordinates, this function contructs a non-directed graph, by conncting adjected points.
    There are three combinations of settings:
        Allow all neigbors:     Distance(a, b) <= sqrt(2)
        Allow only manhattan:   Distance(a, b) == 1
        Allow only euclidean:   Distance(a, b) == sqrt(2)


    :param coordiniates_or_tiles: A set of coordinates.
    :type coordiniates_or_tiles: Tiles
    :param allow_euclidean_connections: Whether to regard diagonal adjected cells as neighbors
    :type: bool
    :param allow_manhattan_connections: Whether to regard directly adjected cells as neighbors
    :type: bool

    :return: A graph with nodes that are conneceted as specified by the parameters.
    :rtype: nx.Graph
    """
    assert allow_euclidean_connections or allow_manhattan_connections
    if hasattr(coordiniates_or_tiles, 'positions'):
        coordiniates_or_tiles = coordiniates_or_tiles.positions
    possible_connections = itertools.combinations(coordiniates_or_tiles, 2)
    graph = nx.Graph()
    for a, b in possible_connections:
        diff = np.linalg.norm(np.asarray(a)-np.asarray(b))
        if allow_manhattan_connections and allow_euclidean_connections and diff <= np.sqrt(2):
            graph.add_edge(a, b)
        elif not allow_manhattan_connections and allow_euclidean_connections and diff == np.sqrt(2):
            graph.add_edge(a, b)
        elif allow_manhattan_connections and not allow_euclidean_connections and diff == 1:
            graph.add_edge(a, b)
    return graph
