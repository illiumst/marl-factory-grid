import numpy as np
from pathlib import Path
from environments import helpers as h


class BaseFactory:
    LEVELS_DIR = 'levels'
    _level_idx = 0
    _agent_start_idx = 1
    _is_free_cell = 0
    _is_occupied_cell = 1

    def __init__(self, level='simple', n_agents=1, max_steps=1e3):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.level = h.one_hot_level(
            h.parse_level(Path(__file__).parent / self.LEVELS_DIR / f'{level}.txt')
        )
        self.slice_strings = {0: 'level', **{i: f'agent#{i}' for i in range(1, self.n_agents+1)}}
        self.reset()

    def reset(self):
        self.done = False
        self.steps = 0
        # Agent placement ...
        agents = np.zeros((self.n_agents, *self.level.shape), dtype=np.int8)
        floor_tiles = np.argwhere(self.level == self._is_free_cell)
        # ... on random positions
        np.random.shuffle(floor_tiles)
        for i, (x, y) in enumerate(floor_tiles[:self.n_agents]):
            agents[i, x, y] = self._is_occupied_cell
        # state.shape = level, agent 1,..., agent n,
        self.state = np.concatenate((np.expand_dims(self.level, axis=0), agents), axis=0)
        # Returns State, Reward, Done, Info
        return self.state, 0, self.done, {}

    def step(self, actions):
        assert type(actions) in [int, list]
        if type(actions) == int:
            actions = [actions]
        self.steps += 1
        r = 0
        collision_vecs = np.zeros((self.n_agents, self.state.shape[0]))  # n_agents x n_slices
        for i, a in enumerate(actions):
            old_pos, new_pos, valid = h.check_agent_move(state=self.state, dim=i+self._agent_start_idx, action=a)
            if valid:  # Does not collide width level boundrys
                self.make_move(i, old_pos, new_pos)
            else:  # Trying to leave the level
                collision_vecs[i, self._level_idx] = self._is_occupied_cell  # Collides with level boundrys

        # For each agent check for abitrary collions:
        for i in range(self.n_agents):  # Note: might as well save the positions (redundant): return value of make_move
            agent_slice = self.state[i+self._agent_start_idx]
            x, y = np.argwhere(agent_slice == self._is_occupied_cell)[0]    # current position of agent i
            collisions_vec = self.state[:, x, y].copy()                     # "vertical fiber" at position of agent i
            collisions_vec[i+self._agent_start_idx] = self._is_free_cell    # no self-collisions
            collision_vecs[i] += collisions_vec
        reward, info = self.step_core(collision_vecs, actions, r)
        r += reward
        if self.steps >= self.max_steps:
            self.done = True
        return self.state, r, self.done, info

    def make_move(self, agent_i, old_pos, new_pos):
        (x, y), (x_new, y_new) = old_pos, new_pos
        self.state[agent_i+self._agent_start_idx, x, y] = self._is_free_cell
        self.state[agent_i+self._agent_start_idx, x_new, y_new] = self._is_occupied_cell
        return new_pos

    @property
    def free_cells(self) -> np.ndarray:
        free_cells = self.state.sum(0)
        free_cells = np.argwhere(free_cells == self._is_free_cell)
        np.random.shuffle(free_cells)
        return free_cells

    def step_core(self, collisions_vec, actions, r):
        # Returns: Reward, Info
        # Set to "raise NotImplementedError"
        return 0, {}  # What is returned here?
