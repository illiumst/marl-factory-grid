from pathlib import Path
from typing import List, Union, Iterable

import gym
import numpy as np
from gym import spaces

import yaml
from gym.wrappers import FrameStack

from environments import helpers as h
from environments.utility_classes import Actions, StateSlices, AgentState, MovementProperties, Zones, DoorState


# noinspection PyAttributeOutsideInit
class BaseFactory(gym.Env):

    @property
    def action_space(self):
        return spaces.Discrete(self._actions.n)

    @property
    def observation_space(self):
        agent_slice = self.n_agents if self.omit_agent_slice_in_obs else 0
        agent_slice = (self.n_agents - 1) if self.combin_agent_slices_in_obs else agent_slice
        if self.pomdp_radius:
            shape = (self._state.shape[0] - agent_slice, self.pomdp_radius * 2 + 1, self.pomdp_radius * 2 + 1)
            space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
            return space
        else:
            shape = [x-agent_slice if idx == 0 else x for idx, x in enumerate(self._state.shape)]
            space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
            return space

    @property
    def movement_actions(self):
        return self._actions.movement_actions

    @property
    def has_doors(self):
        return hasattr(self, '_doors')

    def __enter__(self):
        return self if self.frames_to_stack == 0 else FrameStack(self, self.frames_to_stack)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, level_name='simple', n_agents=1, max_steps=int(5e2), pomdp_radius: Union[None, int] = 0,
                 movement_properties: MovementProperties = MovementProperties(), parse_doors=False,
                 combin_agent_slices_in_obs: bool = False, frames_to_stack=0,
                 omit_agent_slice_in_obs=False, **kwargs):
        assert (combin_agent_slices_in_obs != omit_agent_slice_in_obs) or \
               (not combin_agent_slices_in_obs and not omit_agent_slice_in_obs), \
            'Both options are exclusive'
        assert frames_to_stack != 1 and frames_to_stack >= 0, "'frames_to_stack' cannot be negative or 1."

        self.movement_properties = movement_properties
        self.level_name = level_name

        self.n_agents = n_agents
        self.max_steps = max_steps
        self.pomdp_radius = pomdp_radius
        self.combin_agent_slices_in_obs = combin_agent_slices_in_obs
        self.omit_agent_slice_in_obs = omit_agent_slice_in_obs
        self.frames_to_stack = frames_to_stack

        self.done_at_collision = False

        self._state_slices = StateSlices()

        # Level
        level_filepath = Path(__file__).parent / h.LEVELS_DIR / f'{self.level_name}.txt'
        parsed_level = h.parse_level(level_filepath)
        self._level = h.one_hot_level(parsed_level)
        level_slices = [h.LEVEL]

        # Doors
        if parse_doors:
            parsed_doors = h.one_hot_level(parsed_level, h.DOOR)
            if parsed_doors.any():
                self._doors = parsed_doors
                level_slices.append(h.DOORS)

        # Agents
        offset = len(level_slices)
        self._state_slices.register_additional_items([*level_slices,
                                                      *[f'agent#{i}' for i in range(offset, n_agents + offset)]])

        # Additional Slices from SubDomains
        if 'additional_slices' in kwargs:
            self._state_slices.register_additional_items(kwargs.get('additional_slices'))
        self._zones = Zones(parsed_level)

        self._actions = Actions(self.movement_properties, can_use_doors=self.has_doors)
        self._actions.register_additional_items(self.additional_actions)
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
        slices = [np.expand_dims(self._level, 0)]
        self._steps = 0
        self._agent_states = list()

        # Door Init
        if self.has_doors:
            self._door_states = [DoorState(i, tuple(pos)) for i, pos
                                 in enumerate(np.argwhere(self._doors == h.IS_OCCUPIED_CELL))]
            slices.append(np.expand_dims(self._doors, 0))

        # Agent placement ...
        floor_tiles = np.argwhere(self._level == h.IS_FREE_CELL)
        # ... on random positions
        np.random.shuffle(floor_tiles)
        agents = np.zeros((self.n_agents, *self._level.shape), dtype=np.int8)
        for i, (x, y) in enumerate(floor_tiles[:self.n_agents]):
            agents[i, x, y] = h.IS_OCCUPIED_CELL
            agent_state = AgentState(i, -1, pos=(x, y))
            self._agent_states.append(agent_state)
        slices.append(agents)

        # GLOBAL STATE
        self._state = np.concatenate(slices, axis=0)
        return None

    def _get_observations(self) -> np.ndarray:
        if self.n_agents == 1:
            obs = self._build_per_agent_obs(0)
        elif self.n_agents >= 2:
            obs = np.stack([self._build_per_agent_obs(agent_i) for agent_i in range(self.n_agents)])
        else:
            raise ValueError('n_agents cannot be smaller than 1!!')
        return obs

    def _build_per_agent_obs(self, agent_i: int) -> np.ndarray:
        first_agent_slice = self._state_slices.AGENTSTARTIDX
        # Todo: make this more efficient!
        if self.pomdp_radius:
            pomdp_diameter = self.pomdp_radius * 2 + 1
            global_x, global_y = self._agent_states[agent_i].pos
            x0, x1 = max(0, global_x - self.pomdp_radius), global_x + self.pomdp_radius + 1
            y0, y1 = max(0, global_y - self.pomdp_radius), global_y + self.pomdp_radius + 1
            obs = self._state[:, x0:x1, y0:y1]
            if obs.shape[1] != pomdp_diameter or obs.shape[2] != pomdp_diameter:
                obs_padded = np.full((obs.shape[0], pomdp_diameter, pomdp_diameter), h.IS_OCCUPIED_CELL)
                local_x, local_y = np.argwhere(obs[first_agent_slice + agent_i] == h.IS_OCCUPIED_CELL)[0]
                obs_padded[:,
                abs(local_x-self.pomdp_radius):abs(local_x-self.pomdp_radius)+obs.shape[1],
                abs(local_y-self.pomdp_radius):abs(local_y-self.pomdp_radius)+obs.shape[2]] = obs
                obs = obs_padded
        else:
            obs = self._state
        if self.omit_agent_slice_in_obs:
            obs_new = obs[[key for key, val in self._state_slices.items() if h.AGENT not in val]]
            return obs_new
        else:
            if self.combin_agent_slices_in_obs:
                agent_obs = np.sum(obs[[key for key, val in self._state_slices.items() if 'agent' in val]],
                                   axis=0, keepdims=True)
                obs = np.concatenate((obs[:first_agent_slice], agent_obs, obs[first_agent_slice+self.n_agents:]))
                return obs
            else:
                return obs

    def do_additional_actions(self, agent_i: int, action: int) -> ((int, int), bool):
        raise NotImplementedError

    def step(self, actions):
        actions = [actions] if isinstance(actions, int) or np.isscalar(actions) else actions
        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self._steps += 1
        done = False

        # Move this in a seperate function?
        for agent_i, action in enumerate(actions):
            agent = self._agent_states[agent_i]
            if self._actions.is_moving_action(action):
                pos, valid = self.move_or_colide(agent_i, action)
            elif self._actions.is_no_op(action):
                pos, valid = agent.pos, h.VALID
            elif self._actions.is_door_usage(action):
                # Check if agent raly stands on a door:
                if self._state[self._state_slices.by_name(h.DOORS)][agent.pos] in [h.IS_OCCUPIED_CELL, ]:
                    door = [door for door in self._door_states if door.pos == self._agent_states[agent_i].pos][0]
                    door.use()
                    pos, valid = self._agent_states[agent_i].pos, h.VALID
                # When he doesn't...
                else:
                    pos, valid = self._agent_states[agent_i].pos, h.NOT_VALID
            else:
                pos, valid = self.do_additional_actions(agent_i, action)
            # Update state accordingly
            self._agent_states[agent_i].update(pos=pos, action_valid=valid, action=action)

        for i, collision_vec in enumerate(self.check_all_collisions(self._agent_states, self._state.shape[0])):
            self._agent_states[i].update(collision_vector=collision_vec)
            if self.done_at_collision and collision_vec.any():
                done = True

        # Step the door close intervall
        agents_pos = [agent.pos for agent in self._agent_states]
        if self.has_doors:
            for door_i, door in enumerate(self._door_states):
                if door.is_open and door.time_to_close and door.pos not in agents_pos:
                    door.time_to_close -= 1
                elif door.is_open and not door.time_to_close and door.pos not in agents_pos:
                    door.use()
                    self._state[self._state_slices.by_name(h.DOORS)] = 1 if door.is_closed else -1

        reward, info = self.calculate_reward(self._agent_states)

        if self._steps >= self.max_steps:
            done = True

        info.update(step_reward=reward, step=self._steps)

        return None, reward, done, info

    def check_all_collisions(self, agent_states: List[AgentState], collisions: int) -> np.ndarray:
        collision_vecs = np.zeros((len(agent_states), collisions))  # n_agents x n_slices
        for agent_state in agent_states:
            # Register only collisions of moving agents
            if self._actions.is_moving_action(agent_state.action):
                collision_vecs[agent_state.i] = self.check_collisions(agent_state)
        return collision_vecs

    def check_collisions(self, agent_state: AgentState) -> np.ndarray:
        pos_x, pos_y = agent_state.pos
        # FixMe: We need to find a way to spare out some dimensions, eg. an info dimension etc... a[?,]
        #  https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing
        collisions_vec = self._state[:, pos_x, pos_y].copy()                 # "vertical fiber" at position of agent i
        collisions_vec[self._state_slices.AGENTSTARTIDX + agent_state.i] = h.IS_FREE_CELL   # no self-collisions
        if 'door' in self._state_slices.values():
            collisions_vec[self._state_slices.by_name('doors')] = h.IS_FREE_CELL            # no door-collisions

        if agent_state.action_valid:
            # All well, no collision.
            # Place a function hook here if needed.
            pass
        else:
            # Place a marker to indicate a collision with the level boundrys
            collisions_vec[self._state_slices.by_name(h.LEVEL)] = h.IS_OCCUPIED_CELL
        return collisions_vec

    def do_move(self, agent_i: int, old_pos: (int, int), new_pos: (int, int)) -> None:
        (x, y), (x_new, y_new) = old_pos, new_pos
        self._state[agent_i + self._state_slices.AGENTSTARTIDX, x, y] = h.IS_FREE_CELL
        self._state[agent_i + self._state_slices.AGENTSTARTIDX, x_new, y_new] = h.IS_OCCUPIED_CELL

    def move_or_colide(self, agent_i: int, action: int) -> ((int, int), bool):
        old_pos, new_pos, valid = self._check_agent_move(agent_i=agent_i, action=self._actions[action])
        if valid:
            # Does not collide width level boundaries
            self.do_move(agent_i, old_pos, new_pos)
            return new_pos, valid
        else:
            # Agent seems to be trying to collide in this step
            return old_pos, valid

    def _check_agent_move(self, agent_i, action: str):
        agent_slice_idx = self._state_slices.AGENTSTARTIDX + agent_i
        agent_slice = self._state[agent_slice_idx]  # horizontal slice from state tensor
        agent_pos = np.argwhere(agent_slice == 1)
        if len(agent_pos) > 1:
            raise AssertionError('Only one agent per slice is allowed.')
        x, y = agent_pos[0]

        # Actions
        x_diff, y_diff = h.ACTIONMAP[action]
        x_new = x + x_diff
        y_new = y + y_diff

        if self.has_doors and self._agent_states[agent_i]._last_pos != (-1, -1):
            door = [door for door in self._door_states if door.pos == (x, y)]
            if door:
                door = door[0]
                if door.is_open:
                    pass
                else:  # door.is_closed:
                    local_door_map = self._state[self._state_slices.by_name(h.LEVEL)][door.pos[0]-1:door.pos[0]+2,
                                                                                      door.pos[1]-1:door.pos[1]+2]
                    local_agent_map = np.zeros_like(local_door_map)
                    local_agent_map[tuple(np.subtract(door.pos, self._agent_states[agent_i]._last_pos))] += 1
                    local_agent_map[tuple(np.subtract(door.pos, (x_new, y_new)))] += 1
                    if np.all(local_door_map == h.HORIZONTAL_DOOR_MAP):
                        # This is a horizontal Door Configuration
                        if np.sum(local_agent_map[0]) >= 2 or np.sum(local_agent_map[-1]) >= 2:
                            # The Agent goes back to where he came from
                            pass
                        else:
                            # The Agent tries to go through a closed door
                            return (x, y), (x, y), h.NOT_VALID
                    else:
                        # This is a vertical Door Configuration
                        if np.sum(local_agent_map[:, 0]) >= 2 or np.sum(local_agent_map[:, -1]) >= 2:
                            # The Agent goes back to where he came from
                            pass
                        else:
                            # The Agent tries to go through a closed door
                            return (x, y), (x, y), h.NOT_VALID
            else:
                pass
        else:
            pass

        valid = h.check_position(self._state[self._state_slices.by_name(h.LEVEL)], (x_new, y_new))

        return (x, y), (x_new, y_new), valid

    def agent_i_position(self, agent_i: int) -> (int, int):
        positions = np.argwhere(self._state[self._state_slices.AGENTSTARTIDX + agent_i] == h.IS_OCCUPIED_CELL)
        assert positions.shape[0] == 1
        pos_x, pos_y = positions[0]  # a.flatten()
        return pos_x, pos_y

    def free_cells(self, excluded_slices: Union[None, List[int], int] = None) -> np.array:

        excluded_slices = excluded_slices or []
        assert isinstance(excluded_slices, (int, list))
        excluded_slices = excluded_slices if isinstance(excluded_slices, list) else [excluded_slices]

        state = self._state

        if excluded_slices:
            # Todo: Is there a cleaner way?
            #  inds = list(range(self._state.shape[0]))
            #  excluded_slices = [inds[x] if x < 0 else x for x in excluded_slices]
            #  state = self._state[[x for x in inds if x not in excluded_slices]]

            # Yes there is!
            bool_array = np.full(self._state.shape[0], True)
            bool_array[excluded_slices] = False
            state = self._state[bool_array]

        free_cells = np.argwhere(state.sum(0) == h.IS_FREE_CELL)
        np.random.shuffle(free_cells)
        return free_cells

    def calculate_reward(self, agent_states: List[AgentState]) -> (int, dict):
        # Returns: Reward, Info
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        # d = {key: val._asdict() if hasattr(val, '_asdict') else val for key, val in self.__dict__.items()
        d = {key: val for key, val in self.__dict__.items() if not key.startswith('_') and not key.startswith('__')}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w') as f:
            yaml.dump(d, f)
            # pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
