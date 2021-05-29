from typing import List, Union, Iterable

import gym
from gym import spaces
import numpy as np
from pathlib import Path

from environments import helpers as h
from environments.logging.monitor import FactoryMonitor


class AgentState:

    def __init__(self, i: int, action: int):
        self.i = i
        self.action = action

        self.collision_vector = None
        self.action_valid = None
        self.pos = None
        self.info = {}

    @property
    def collisions(self):
        return np.argwhere(self.collision_vector != 0).flatten()

    def update(self, **kwargs):                             # is this hacky?? o.0
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                raise AttributeError(f'"{key}" cannot be updated, this attr is not a part of {self.__class__.__name__}')


class Register:

    @property
    def n(self):
        return len(self)

    def __init__(self):
        self._register = dict()

    def __len__(self):
        return len(self._register)

    def __add__(self, other: Union[str, List[str]]):
        other = other if isinstance(other, list) else [other]
        assert all([isinstance(x, str) for x in other]), f'All item names have to be of type {str}.'
        self._register.update({key+len(self._register): value for key, value in enumerate(other)})
        return self

    def register_additional_items(self, other: Union[str, List[str]]):
        self_with_additional_items = self + other
        return self_with_additional_items

    def __getitem__(self, item):
        return self._register[item]

    def by_name(self, item):
        return list(self._register.keys())[list(self._register.values()).index(item)]

    def __repr__(self):
        return f'{self.__class__.__name__}({self._register})'


class Actions(Register):

    @property
    def movement_actions(self):
        return self._movement_actions

    def __init__(self, allow_square_movement=False, allow_diagonal_movement=False, allow_no_op=False):
        # FIXME: There is a bug in helpers because there actions are ints. and the order matters.
        assert not(allow_square_movement is False and allow_diagonal_movement is True), "There is a bug in helpers!!!"
        super(Actions, self).__init__()
        self.allow_no_op = allow_no_op
        self.allow_diagonal_movement = allow_diagonal_movement
        self.allow_square_movement = allow_square_movement
        if allow_square_movement:
            self + ['north', 'east', 'south', 'west']
        if allow_diagonal_movement:
            self + ['north-east', 'south-east', 'south-west', 'north-west']
        self._movement_actions = self._register.copy()
        if self.allow_no_op:
            self + 'no-op'


class StateSlice(Register):

    def __init__(self, n_agents: int):
        super(StateSlice, self).__init__()
        offset = 1
        self.register_additional_items(['level', *[f'agent#{i}' for i in range(offset, n_agents+offset)]])


class BaseFactory(gym.Env):

    @property
    def action_space(self):
        return spaces.Discrete(self._actions.n)

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1, shape=self.state.shape, dtype=np.float32)

    @property
    def movement_actions(self):
        return self._actions.movement_actions


    def __init__(self, level='simple', n_agents=1, max_steps=int(2e2), **kwargs):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.done_at_collision = False
        _actions = Actions(allow_square_movement=kwargs.get('allow_square_movement', True),
                           allow_diagonal_movement=kwargs.get('allow_diagonal_movement', True),
                           allow_no_op=kwargs.get('allow_no_op', True))
        self._actions = _actions + self.additional_actions

        self.level = h.one_hot_level(
            h.parse_level(Path(__file__).parent / h.LEVELS_DIR / f'{level}.txt')
        )
        self.state_slices = StateSlice(n_agents)
        self.reset()

    @property
    def additional_actions(self) -> Union[str, List[str]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!
        Please return a dict with the given types -> {int: str}.
        The int should start at 0.

        :return:            An Actions-object holding all actions with keys in range 0-n.
        :rtype:             Actions
        """
        raise NotImplementedError('Please register additional actions ')

    def reset(self) -> (np.ndarray, int, bool, dict):
        self.steps = 0
        self.monitor = FactoryMonitor(self)
        self.agent_states = []
        # Agent placement ...
        agents = np.zeros((self.n_agents, *self.level.shape), dtype=np.int8)
        floor_tiles = np.argwhere(self.level == h.IS_FREE_CELL)
        # ... on random positions
        np.random.shuffle(floor_tiles)
        for i, (x, y) in enumerate(floor_tiles[:self.n_agents]):
            agents[i, x, y] = h.IS_OCCUPIED_CELL
            agent_state = AgentState(i, -1)
            agent_state.update(pos=[x, y])
            self.agent_states.append(agent_state)
        # state.shape = level, agent 1,..., agent n,
        self.state = np.concatenate((np.expand_dims(self.level, axis=0), agents), axis=0)
        # Returns State
        return self.state

    def do_additional_actions(self, agent_i: int, action: int) -> ((int, int), bool):
        raise NotImplementedError

    def step(self, actions):
        actions = [actions] if isinstance(actions, int) or np.isscalar(actions) else actions
        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self.steps += 1
        done = False

        # Move this in a seperate function?
        agent_states = list()
        for agent_i, action in enumerate(actions):
            agent_i_state = AgentState(agent_i, action)
            if self._is_moving_action(action):
                pos, valid = self.move_or_colide(agent_i, action)
            elif self._is_no_op(action):
                pos, valid = self.agent_i_position(agent_i), True
            else:
                pos, valid = self.do_additional_actions(agent_i, action)
            # Update state accordingly
            agent_i_state.update(pos=pos, action_valid=valid)
            agent_states.append(agent_i_state)

        for i, collision_vec in enumerate(self.check_all_collisions(agent_states, self.state.shape[0])):
            agent_states[i].update(collision_vector=collision_vec)
            if self.done_at_collision and collision_vec.any():
                done = True

        self.agent_states = agent_states
        reward, info = self.calculate_reward(agent_states)

        if self.steps >= self.max_steps:
            done = True
        self.monitor.set('step_reward', reward)
        return self.state, reward, done, info

    def _is_moving_action(self, action):
        return action in self._actions.movement_actions

    def _is_no_op(self, action):
        return self._actions[action] == 'no-op'

    def check_all_collisions(self, agent_states: List[AgentState], collisions: int) -> np.ndarray:
        collision_vecs = np.zeros((len(agent_states), collisions))  # n_agents x n_slices
        for agent_state in agent_states:
            # Register only collisions of moving agents
            if self._is_moving_action(agent_state.action):
                collision_vecs[agent_state.i] = self.check_collisions(agent_state)
        return collision_vecs

    def check_collisions(self, agent_state: AgentState) -> np.ndarray:
        pos_x, pos_y = agent_state.pos
        # FixMe: We need to find a way to spare out some dimensions, eg. an info dimension etc... a[?,]
        collisions_vec = self.state[:, pos_x, pos_y].copy()                 # "vertical fiber" at position of agent i
        collisions_vec[h.AGENT_START_IDX + agent_state.i] = h.IS_FREE_CELL  # no self-collisions
        if agent_state.action_valid:
            # ToDo: Place a function hook here
            pass
        else:
            # Place a marker to indicate a collision with the level boundrys
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
            # Does not collide width level boundaries
            self.do_move(agent_i, old_pos, new_pos)
            return new_pos, valid
        else:
            # Agent seems to be trying to collide in this step
            return old_pos, valid

    def agent_i_position(self, agent_i: int) -> (int, int):
        positions = np.argwhere(self.state[h.AGENT_START_IDX+agent_i] == h.IS_OCCUPIED_CELL)
        assert positions.shape[0] == 1
        pos_x, pos_y = positions[0]  # a.flatten()
        return pos_x, pos_y

    def free_cells(self, excluded_slices: Union[None, List[int], int] = None) -> np.array:
        excluded_slices = excluded_slices or []
        assert isinstance(excluded_slices, (int, list))
        excluded_slices = excluded_slices if isinstance(excluded_slices, list) else [excluded_slices]

        state = self.state

        if excluded_slices:
            # Todo: Is there a cleaner way?
            inds = list(range(self.state.shape[0]))
            excluded_slices = [inds[x] if x < 0 else x for x in excluded_slices]
            state = self.state[[x for x in inds if x not in excluded_slices]]

        free_cells = np.argwhere(state.sum(0) == h.IS_FREE_CELL)
        np.random.shuffle(free_cells)
        return free_cells

    def calculate_reward(self, agent_states: List[AgentState]) -> (int, dict):
        # Returns: Reward, Info
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
