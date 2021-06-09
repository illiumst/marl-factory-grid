from collections import OrderedDict
from typing import List, Union, NamedTuple
import random

import numpy as np

from environments.factory.base_factory import BaseFactory
from environments import helpers as h

from environments.factory.renderer import Renderer, Entity
from environments.utility_classes import AgentState, MovementProperties

DIRT_INDEX = -1
CLEAN_UP_ACTION = 'clean_up'


class DirtProperties(NamedTuple):
    clean_amount: int = 2            # How much does the robot clean with one action.
    max_spawn_ratio: float = 0.2       # On max how much tiles does the dirt spawn in percent.
    gain_amount: float = 0.5           # How much dirt does spawn per tile
    spawn_frequency: int = 5         # Spawn Frequency in Steps
    max_local_amount: int = 1        # Max dirt amount per tile.
    max_global_amount: int = 20      # Max dirt amount in the whole environment.


# noinspection PyAttributeOutsideInit
class SimpleFactory(BaseFactory):

    @property
    def additional_actions(self) -> Union[str, List[str]]:
        return CLEAN_UP_ACTION

    def _is_clean_up_action(self, action: Union[str, int]):
        if isinstance(action, str):
            action = self._actions.by_name(action)
        return self._actions[action] == CLEAN_UP_ACTION

    def __init__(self, *args, dirt_properties: DirtProperties, verbose=False, **kwargs):
        self.dirt_properties = dirt_properties
        self.verbose = verbose
        self.max_dirt = 20
        super(SimpleFactory, self).__init__(*args, additional_slices='dirt', **kwargs)
        self._renderer = None  # expensive - don't use it when not required !

    def render(self):

        if not self._renderer:  # lazy init
            height, width = self._state.shape[1:]
            self._renderer = Renderer(width, height, view_radius=self.pomdp_radius)

        dirt = [Entity('dirt', [x, y], min(0.15 + self._state[DIRT_INDEX, x, y], 1.5), 'scale')
                for x, y in np.argwhere(self._state[DIRT_INDEX] > h.IS_FREE_CELL)]
        walls = [Entity('wall', pos) for pos in np.argwhere(self._state[h.LEVEL_IDX] > h.IS_FREE_CELL)]

        def asset_str(agent):
            if any([x is None for x in [self._state_slices[j] for j in agent.collisions]]):
                print('error')
            cols = ' '.join([self._state_slices[j] for j in agent.collisions])
            if 'agent' in cols:
                return 'agent_collision', 'blank'
            elif not agent.action_valid or 'level' in cols or 'agent' in cols:
                return f'agent{agent.i + 1}', 'invalid'
            elif self._is_clean_up_action(agent.action):
                return f'agent{agent.i + 1}', 'valid'
            else:
                return f'agent{agent.i + 1}', 'idle'
        agents = []
        for i, agent in enumerate(self._agent_states):
            name, state = asset_str(agent)
            agents.append(Entity(name, agent.pos, 1, 'none', state))
        self._renderer.render(dirt+walls+agents)

    def spawn_dirt(self) -> None:
        if not np.argwhere(self._state[DIRT_INDEX] != h.IS_FREE_CELL).shape[0] > self.dirt_properties.max_global_amount:
            free_for_dirt = self.free_cells(excluded_slices=DIRT_INDEX)

            # randomly distribute dirt across the grid
            n_dirt_tiles = int(random.uniform(0, self.dirt_properties.max_spawn_ratio) * len(free_for_dirt))
            for x, y in free_for_dirt[:n_dirt_tiles]:
                new_value = self._state[DIRT_INDEX, x, y] + self.dirt_properties.gain_amount
                self._state[DIRT_INDEX, x, y] = max(new_value, self.dirt_properties.max_local_amount)

        else:
            pass

    def clean_up(self, pos: (int, int)) -> ((int, int), bool):
        new_dirt_amount = self._state[DIRT_INDEX][pos] - self.dirt_properties.clean_amount
        cleanup_was_sucessfull: bool
        if self._state[DIRT_INDEX][pos] == h.IS_FREE_CELL:
            cleanup_was_sucessfull = False
            return pos, cleanup_was_sucessfull
        else:
            cleanup_was_sucessfull = True
            self._state[DIRT_INDEX][pos] = max(new_dirt_amount, h.IS_FREE_CELL)
            return pos, cleanup_was_sucessfull

    def step(self, actions):
        _, reward, done, info = super(SimpleFactory, self).step(actions)
        if not self._next_dirt_spawn:
            self.spawn_dirt()
            self._next_dirt_spawn = self.dirt_properties.spawn_frequency
        else:
            self._next_dirt_spawn -= 1

        obs = self._get_observations()
        return obs, reward, done, info

    def do_additional_actions(self, agent_i: int, action: int) -> ((int, int), bool):
        if action != self._actions.is_moving_action(action):
            if self._is_clean_up_action(action):
                agent_i_pos = self.agent_i_position(agent_i)
                _, valid = self.clean_up(agent_i_pos)
                return agent_i_pos, valid
            else:
                raise RuntimeError('This should not happen!!!')
        else:
            raise RuntimeError('This should not happen!!!')

    def reset(self) -> (np.ndarray, int, bool, dict):
        _ = super().reset()  # state, reward, done, info ... =
        dirt_slice = np.zeros((1, *self._state.shape[1:]))
        self._state = np.concatenate((self._state, dirt_slice))  # dirt is now the last slice
        self.spawn_dirt()
        self._next_dirt_spawn = self.dirt_properties.spawn_frequency
        obs = self._get_observations()
        return obs

    def calculate_reward(self, agent_states: List[AgentState]) -> (int, dict):
        # TODO: What reward to use?
        current_dirt_amount = self._state[DIRT_INDEX].sum()
        dirty_tiles = np.argwhere(self._state[DIRT_INDEX] != h.IS_FREE_CELL).shape[0]
        info_dict = dict()

        try:
            # penalty = current_dirt_amount
            reward = 0
        except (ZeroDivisionError, RuntimeWarning):
            reward = 0

        for agent_state in agent_states:
            cols = agent_state.collisions

            list_of_collisions = [self._state_slices[entity] for entity in cols
                                  if entity != self._state_slices.by_name("dirt")]

            if list_of_collisions:
                self.print(f't = {self._steps}\tAgent {agent_state.i} has collisions with '
                           f'{list_of_collisions}')

            if self._is_clean_up_action(agent_state.action):
                if agent_state.action_valid:
                    reward += 1
                    self.print(f'Agent {agent_state.i} did just clean up some dirt at {agent_state.pos}.')
                    info_dict.update(dirt_cleaned=1)
                else:
                    reward -= 0.01
                    self.print(f'Agent {agent_state.i} just tried to clean up some dirt '
                               f'at {agent_state.pos}, but was unsucsessfull.')
                    info_dict.update(failed_cleanup_attempt=1)

            elif self._actions.is_moving_action(agent_state.action):
                if agent_state.action_valid:
                    # info_dict.update(movement=1)
                    reward -= 0.00
                else:
                    # info_dict.update(collision=1)
                    # self.print('collision')
                    reward -= 0.01

            else:
                info_dict.update(no_op=1)
                reward -= 0.00

            for entity in list_of_collisions:
                info_dict.update({f'agent_{agent_state.i}_vs_{entity}': 1})

        info_dict.update(dirt_amount=current_dirt_amount)
        info_dict.update(dirty_tile_count=dirty_tiles)
        self.print(f"reward is {reward}")
        # Potential based rewards ->
        #  track the last reward , minus the current reward = potential
        return reward, info_dict

    def print(self, string):
        if self.verbose:
            print(string)


if __name__ == '__main__':
    render = True

    move_props = MovementProperties(allow_diagonal_movement=True, allow_square_movement=True)
    dirt_props = DirtProperties()
    factory = SimpleFactory(movement_properties=move_props, dirt_properties=dirt_props, n_agents=2,
                            combin_agent_slices_in_obs=True, omit_agent_slice_in_obs=False)

    # dirt_props = DirtProperties()
    # move_props = MovementProperties(allow_diagonal_movement=False, allow_no_op=False)
    # factory = SimpleFactory(n_agents=2, dirt_properties=dirt_props, movement_properties=move_props, level='rooms',
    #                         pomdp_radius=2)

    n_actions = factory.action_space.n - 1

    for epoch in range(100):
        random_actions = [[random.randint(0, n_actions) for _ in range(factory.n_agents)] for _ in range(200)]
        env_state = factory.reset()
        r = 0
        for agent_i_action in random_actions:
            env_state, step_r, done_bool, info_obj = factory.step(agent_i_action)
            r += step_r
            if render:
                factory.render()
            if done_bool:
                break
        print(f'Factory run {epoch} done, reward is:\n    {r}')
