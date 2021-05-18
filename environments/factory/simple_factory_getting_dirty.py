from collections import OrderedDict
from dataclasses import dataclass
from typing import List
import random

import numpy as np

from environments.factory.base_factory import BaseFactory, AgentState
from environments import helpers as h

from environments.factory.renderer import Renderer
from environments.factory.renderer import Entity

DIRT_INDEX = -1


@dataclass
class DirtProperties:
    clean_amount = 0.25
    max_spawn_ratio = 0.1
    gain_amount = 0.1
    spawn_frequency = 5


class GettingDirty(BaseFactory):

    def _is_clean_up_action(self, action):
        # Account for NoOP; remove -1 when activating NoOP
        return self.movement_actions + 1 - 1 == action

    def __init__(self, *args, dirt_properties: DirtProperties, **kwargs):
        self._dirt_properties = dirt_properties
        super(GettingDirty, self).__init__(*args, **kwargs)
        self.slice_strings.update({self.state.shape[0]-1: 'dirt'})
        self.renderer = None  # expensive - dont use it when not required !

    def render(self):
        if not self.renderer:  # lazy init
            height, width = self.state.shape[1:]
            self.renderer = Renderer(width, height, view_radius=0)

        dirt   = [Entity('dirt', [x, y], min(self.state[DIRT_INDEX, x, y], 1), 'scale')
                  for x, y in np.argwhere(self.state[DIRT_INDEX] > h.IS_FREE_CELL)]
        walls  = [Entity('dirt', pos) for pos in np.argwhere(self.state[h.LEVEL_IDX] > h.IS_FREE_CELL)]
        agents = [Entity('agent', pos) for pos in np.argwhere(self.state[h.AGENT_START_IDX] > h.IS_FREE_CELL)]

        self.renderer.render(OrderedDict(dirt=dirt, wall=walls, agent=agents))

    def spawn_dirt(self) -> None:
        free_for_dirt = self.free_cells(excluded_slices=DIRT_INDEX)

        # randomly distribute dirt across the grid
        n_dirt_tiles = int(random.uniform(0, self._dirt_properties.max_spawn_ratio) * len(free_for_dirt))
        for x, y in free_for_dirt[:n_dirt_tiles]:
            self.state[DIRT_INDEX, x, y] += self._dirt_properties.gain_amount

    def clean_up(self, pos: (int, int)) -> ((int, int), bool):
        new_dirt_amount = self.state[DIRT_INDEX][pos] - self._dirt_properties.clean_amount
        cleanup_was_sucessfull: bool
        if self.state[DIRT_INDEX][pos] == h.IS_FREE_CELL:
            cleanup_was_sucessfull = False
            return pos, cleanup_was_sucessfull
        else:
            cleanup_was_sucessfull = True
            self.state[DIRT_INDEX][pos] = max(new_dirt_amount, h.IS_FREE_CELL)
            return pos, cleanup_was_sucessfull

    def step(self, actions):
        _, _, _, info = super(GettingDirty, self).step(actions)
        if not self.next_dirt_spawn:
            self.spawn_dirt()
            self.next_dirt_spawn = self._dirt_properties.spawn_frequency
        else:
            self.next_dirt_spawn -= 1
        return self.state, self.cumulative_reward, self.done, info

    def additional_actions(self, agent_i: int, action: int) -> ((int, int), bool):
        if action != self._is_moving_action(action):
            if self._is_clean_up_action(action):
                agent_i_pos = self.agent_i_position(agent_i)
                _, valid = self.clean_up(agent_i_pos)
                if valid:
                    print(f'Agent {agent_i} did just clean up some dirt at {agent_i_pos}.')
                    self.monitor.add('dirt_cleaned', self._dirt_properties.clean_amount)
                else:
                    print(f'Agent {agent_i} just tried to clean up some dirt at {agent_i_pos}, but was unsucsessfull.')
                    self.monitor.add('failed_cleanup_attempt', 1)
                return agent_i_pos, valid
            else:
                raise RuntimeError('This should not happen!!!')
        else:
            raise RuntimeError('This should not happen!!!')

    def reset(self) -> (np.ndarray, int, bool, dict):
        state, r, done, _ = super().reset()  # state, reward, done, info ... =
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        self.spawn_dirt()
        self.next_dirt_spawn = self._dirt_properties.spawn_frequency
        return self.state, r, self.done, {}

    def calculate_reward(self, agent_states: List[AgentState]) -> (int, dict):
        # TODO: What reward to use?
        current_dirt_amount = self.state[DIRT_INDEX].sum()
        dirty_tiles = len(np.nonzero(self.state[DIRT_INDEX]))

        this_step_reward = -(dirty_tiles / current_dirt_amount)

        for agent_state in agent_states:
            collisions = agent_state.collisions
            print(f't = {self.steps}\tAgent {agent_state.i} has collisions with '
                  f'{[self.slice_strings[entity] for entity in collisions if entity != self.string_slices["dirt"]]}')
            if self._is_clean_up_action(agent_state.action) and agent_state.action_valid:
                this_step_reward += 1

            for entity in collisions:
                if entity != self.string_slices["dirt"]:
                    self.monitor.add(f'agent_{agent_state.i}_vs_{self.slice_strings[entity]}', 1)
        self.monitor.set('dirt_amount', current_dirt_amount)
        self.monitor.set('dirty_tiles', dirty_tiles)
        return this_step_reward, {}


if __name__ == '__main__':
    render = True

    dirt_props = DirtProperties()
    factory = GettingDirty(n_agents=1, dirt_properties=dirt_props)
    monitor_list = list()
    for epoch in range(100):
        random_actions = [random.randint(0, 8) for _ in range(200)]
        env_state, reward, done_bool, _ = factory.reset()
        for agent_i_action in random_actions:
            env_state, reward, done_bool, info_obj = factory.step(agent_i_action)
            if render:
                factory.render()
        monitor_list.append(factory.monitor.to_pd_dataframe())
        print(f'Factory run {epoch} done, reward is:\n    {reward}')

    from pathlib import Path
    import pickle
    out_path = Path('debug_out')
    out_path.mkdir(exist_ok=True, parents=True)
    with (out_path / 'monitor.pick').open('wb') as f:
        pickle.dump(monitor_list, f, protocol=pickle.HIGHEST_PROTOCOL)
