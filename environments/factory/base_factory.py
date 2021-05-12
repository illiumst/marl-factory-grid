import random
from typing import Tuple, List, Union, Iterable

import numpy as np
from pathlib import Path
from environments import helpers as h


class BaseFactory:

    def __init__(self, level='simple', n_agents=1, max_steps=1e3):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.allow_vertical_movement = True
        self.allow_horizontal_movement = True
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

        # FixMe: Why do we need this?
        r = 0

        # Move this in a seperate function?
        actions = list(enumerate(actions))
        random.shuffle(actions)
        for agent_i, action in actions:
            if self._is_moving_action(action):
                pos, valid = self.move_or_colide(agent_i, action)
            else:
                pos, valid = self.additional_actions(agent_i, action)
            actions[agent_i] = (agent_i, action, pos, valid)

        collision_vecs = self.check_all_collisions(actions, self.state.shape[0])

        reward, info = self.calculate_reward(collision_vecs, [a[1] for a in actions], r)
        r += reward
        if self.steps >= self.max_steps:
            self.done = True
        return self.state, r, self.done, info

    def _is_moving_action(self, action):
        movement_actions = (int(self.allow_vertical_movement) + int(self.allow_horizontal_movement)) * 4
        if action < movement_actions:
            return True
        else:
            return False

    def check_all_collisions(self, agent_action_pos_valid_tuples: (int, int, (int, int), bool), collisions: int) -> np.ndarray:
        collision_vecs = np.zeros((len(agent_action_pos_valid_tuples), collisions))  # n_agents x n_slices
        for agent_i, action, pos, valid in agent_action_pos_valid_tuples:
            if self._is_moving_action(action):
                collision_vecs[agent_i] = self.check_collisions(agent_i, pos, valid)
        return collision_vecs

    def check_collisions(self, agent_i: int, pos: (int, int), valid: bool) -> np.ndarray:
        pos_x, pos_y = pos
        # FixMe: We need to find a way to spare out some dimensions, eg. an info dimension etc... a[?,]
        collisions_vec = self.state[:, pos_x, pos_y].copy()           # "vertical fiber" at position of agent i
        collisions_vec[h.AGENT_START_IDX + agent_i] = h.IS_FREE_CELL  # no self-collisions
        if valid:
            # ToDo: Place a function hook here
            pass
        else:
            collisions_vec[h.LEVEL_IDX] = h.IS_OCCUPIED_CELL
        return collisions_vec

    def do_move(self, agent_i: int, old_pos: (int, int), new_pos: (int, int)) -> None:
        (x, y), (x_new, y_new) = old_pos, new_pos
        self.state[agent_i + h.AGENT_START_IDX, x, y] = h.IS_FREE_CELL
        self.state[agent_i + h.AGENT_START_IDX, x_new, y_new] = h.IS_OCCUPIED_CELL

    def move_or_colide(self, agent_i: int, action: int) -> ((int, int), bool):
        old_pos, new_pos, valid = h.check_agent_move(state=self.state,
                                                     dim=agent_i + h.AGENT_START_IDX,
                                                     action=action)
        if valid:
            # Does not collide width level boundrys
            self.do_move(agent_i, old_pos, new_pos)
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

    def calculate_reward(self, collisions_vec: np.ndarray, actions: Iterable[int], r: int) -> (int, dict):
        # Returns: Reward, Info
        # Set to "raise NotImplementedError"
        return 0, {}
