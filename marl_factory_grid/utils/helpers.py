import importlib

from collections import defaultdict
from pathlib import PurePath, Path
from typing import Union, Dict, List

import numpy as np
from numpy.typing import ArrayLike

from marl_factory_grid.environment import constants as c

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


LEVELS_DIR = 'modules/levels'                    # for use in studies and experiments
STEPS_START = 1                                  # Define where to the stepcount; which is the first step

# Not used anymore? Clean!
# TO_BE_AVERAGED = ['dirt_amount', 'dirty_tiles']
IGNORED_DF_COLUMNS = ['Episode', 'Run',          # For plotting, which values are ignored when loading monitor files
                      'train_step', 'step', 'index', 'dirt_amount', 'dirty_tile_count', 'terminal_observation',
                      'episode']

POS_MASK = np.asarray([[[-1, -1], [0, -1], [1, -1]],
                       [[-1,  0], [0,  0], [1,  0]],
                       [[-1,  1], [0,  1], [1,  1]]])

MOVEMAP = defaultdict(lambda:   (0, 0),
                      {c.NORTH: (-1, 0), c.NORTHEAST: (-1, 1),
                       c.EAST:  (0, 1),  c.SOUTHEAST: (1, 1),
                       c.SOUTH: (1, 0),  c.SOUTHWEST: (1, -1),
                       c.WEST:  (0, -1), c.NORTHWEST: (-1, -1)
                       }
                      )


class ObservationTranslator:

    def __init__(self, this_named_observation_space: Dict[str, dict],
                 *per_agent_named_obs_spaces: Dict[str, dict],
                 placeholder_fill_value: Union[int, str, None] = None):
        """
        This is a helper class, which converts agent observations from joined environments.
        For example, agent trained in different environments may expect different observations.
        This class translates from larger observations spaces to smaller.
        A string _identifier based approach is used.
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

    def translate_observation(self, agent_idx: int, obs):
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
        This is a helper class, which converts agent action spaces to a joined environments action space.
        For example, agent trained in different environments may have different action spaces.
        This class translates from smaller individual agent action spaces to larger joined spaces.
        A string _identifier based approach is used.

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


def one_hot_level(level, symbol: str):
    """
    Given a string based level representation (list of lists, see function `parse_level`), this function creates a
    binary numpy array or `grid`. Grid values that equal `wall_char` become of `Constants.OCCUPIED_CELL` value.
    Can be changed to filter for any symbol.

    :param level: String based level representation (list of lists, see function `parse_level`).
    :param symbol: List[List[str]]

    :return: Binary numpy array
    :rtype: np.typing._array_like.ArrayLike
    """

    grid = np.array(level)
    binary_grid = np.zeros(grid.shape, dtype=np.int8)
    binary_grid[grid == str(symbol)] = c.VALUE_OCCUPIED_CELL
    return binary_grid


def is_move(action_name: str):
    return action_name in MOVEMAP.keys()


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
        if any(c.AGENT in name for name in col_names):
            return 'agent_collision', 'blank'
        elif not valid or c.LEVEL in col_names or c.AGENT in col_names:
            return c.AGENT, 'invalid'
        elif valid and not is_move(action):
            return c.AGENT, 'valid'
        elif valid and is_move(action):
            return c.AGENT, 'move'
        else:
            return c.AGENT, 'idle'
    else:
        return c.AGENT, 'idle'


def locate_and_import_class(class_name, folder_path: Union[str, PurePath] = ''):
    """Locate an object by name or dotted path, importing as necessary."""
    import sys
    sys.path.append("../../environment")
    folder_path = Path(folder_path).resolve()
    module_paths = [x.resolve() for x in folder_path.rglob('*.py') if x.is_file() and '__init__' not in x.name]
    # possible_package_path = folder_path / '__init__.py'
    # package = str(possible_package_path) if possible_package_path.exists() else None
    all_found_modules = list()
    package_pos = next(idx for idx, x in enumerate(Path(__file__).resolve().parts) if x == 'marl_factory_grid')
    for module_path in module_paths:
        module_parts = [x.replace('.py', '') for idx, x in enumerate(module_path.parts) if idx >= package_pos]
        mod = importlib.import_module('.'.join(module_parts))
        all_found_modules.extend([x for x in dir(mod) if (not(x.startswith('__') or len(x) <= 2) and x.istitle())
                                  and x not in ['Entity',  'NamedTuple', 'List', 'Rule', 'Union', 'Floor'
                                                'TickResult', 'ActionResult', 'Action', 'Agent', 'BoundEntityMixin',
                                                'RenderEntity', 'TemplateRule', 'Objects', 'PositionMixin',
                                                'IsBoundMixin', 'EnvObject', 'EnvObjects', 'Dict', 'Any'
                                                ]])
        try:
            model_class = mod.__getattribute__(class_name)
            return model_class
        except AttributeError:
            continue
    raise AttributeError(f'Class "{class_name}" was not found in "{folder_path.name}"', list(set(all_found_modules)))
