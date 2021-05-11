import random

import numpy as np
from pathlib import Path
from environments import helpers as h


class BaseFactory:

    def __init__(self, level='simple', n_agents=1, max_steps=1e3):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.level = h.one_hot_level(
            h.parse_level(Path(__file__).parent / h.LEVELS_DIR / f'{level}.txt')
        )
        self.slice_strings = {0: 'level', **{i: f'agent#{i}' for i in range(1, self.n_agents+1)}}
        self.reset()

    def reset(self):
        self.done = False
        self.steps = 0
        # Agent placement ...
        agents = np.zeros((self.n_agents, *self.level.shape), dtype=np.int8)
        floor_tiles = np.argwhere(self.level == h.IS_FREE_CELL)
        # ... on random positions
        np.random.shuffle(floor_tiles)
        for i, (x, y) in enumerate(floor_tiles[:self.n_agents]):
            agents[i, x, y] = h.IS_OCCUPIED_CELL
        # state.shape = level, agent 1,..., agent n,
        self.state = np.concatenate((np.expand_dims(self.level, axis=0), agents), axis=0)
        # Returns State, Reward, Done, Info
        return self.state, 0, self.done, {}

    def additional_actions(self, agent_i, action) -> ((int, int), bool):
        raise NotImplementedError

    def step(self, actions):
        actions = [actions] if isinstance(actions, int) else actions
        assert isinstance(actions, list), f'"actions has to be in [{int, list}]'
        self.steps += 1
        r = 0

        actions = list(enumerate(actions))
        random.shuffle(actions)
        for agent_i, action in actions:
            if action <= 8:
                pos, did_collide = self.move_or_colide(agent_i, action)
            else:
                pos, did_collide = self.additional_actions(agent_i, action)
            actions[agent_i] = (pos, did_collide)

        collision_vecs = np.zeros((self.n_agents, self.state.shape[0]))  # n_agents x n_slices
        for agent_i, action in enumerate(actions):
            collision_vecs[agent_i] = self.check_collisions(agent_i, *action)

        reward, info = self.step_core(collision_vecs, actions, r)
        r += reward
        if self.steps >= self.max_steps:
            self.done = True
        return self.state, r, self.done, info

    def check_collisions(self, agent_i, pos, valid):
        pos_x, pos_y = pos
        collisions_vec = self.state[:, pos_x, pos_y].copy()  # "vertical fiber" at position of agent i
        collisions_vec[h.AGENT_START_IDX + agent_i] = h.IS_FREE_CELL  # no self-collisions
        if valid:
            pass
        else:
            collisions_vec[h.LEVEL_IDX] = h.IS_OCCUPIED_CELL
        return collisions_vec

    def move(self, agent_i, old_pos, new_pos):
        (x, y), (x_new, y_new) = old_pos, new_pos
        self.state[agent_i + h.AGENT_START_IDX, x, y] = h.IS_FREE_CELL
        self.state[agent_i + h.AGENT_START_IDX, x_new, y_new] = h.IS_OCCUPIED_CELL

    def move_or_colide(self, agent_i, action) -> ((int, int), bool):
        old_pos, new_pos, valid = h.check_agent_move(state=self.state,
                                                           dim=agent_i + h.AGENT_START_IDX,
                                                           action=action)
        if valid:
            # Does not collide width level boundrys
            self.move(agent_i, old_pos, new_pos)
            return new_pos, valid
        else:
            # Agent seems to be trying to collide in this step
            return old_pos, valid

    @property
    def free_cells(self) -> np.ndarray:
        free_cells = self.state.sum(0)
        free_cells = np.argwhere(free_cells == h.IS_FREE_CELL)
        np.random.shuffle(free_cells)
        return free_cells

    def step_core(self, collisions_vec, actions, r):
        # Returns: Reward, Info
        # Set to "raise NotImplementedError"
        return 0, {}  # What is returned here?
