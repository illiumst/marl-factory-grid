import abc
import time
from enum import Enum
from pathlib import Path
from typing import List, Union, Iterable, Dict
import numpy as np

import gym
from gym import spaces
from gym.wrappers import FrameStack

from environments.factory.base.shadow_casting import Map
from environments.factory.renderer import Renderer, RenderEntity
from environments.helpers import Constants as c, Constants
from environments import helpers as h
from environments.factory.base.objects import Agent, Tile, Action
from environments.factory.base.registers import Actions, Entities, Agents, Doors, FloorTiles, WallTiles, PlaceHolders
from environments.utility_classes import MovementProperties

import simplejson


REC_TAC = 'rec_'


# noinspection PyAttributeOutsideInit
class BaseFactory(gym.Env):

    @property
    def action_space(self):
        return spaces.Discrete(len(self._actions))

    @property
    def observation_space(self):
        if r := self.pomdp_r:
            z = self._obs_cube.shape[0]
            xy = r*2 + 1
            level_shape = (z, xy, xy)
        else:
            level_shape = self._obs_cube.shape
        space = spaces.Box(low=0, high=1, shape=level_shape, dtype=np.float32)
        return space

    @property
    def pomdp_diameter(self):
        return self.pomdp_r * 2 + 1

    @property
    def movement_actions(self):
        return self._actions.movement_actions

    def __enter__(self):
        return self if self.frames_to_stack == 0 else FrameStack(self, self.frames_to_stack)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, level_name='simple', n_agents=1, max_steps=int(5e2), pomdp_r: Union[None, int] = 0,
                 movement_properties: MovementProperties = MovementProperties(), parse_doors=False,
                 combin_agent_obs: bool = False, frames_to_stack=0, record_episodes=False,
                 omit_agent_in_obs=False, done_at_collision=False, cast_shadows=True, additional_agent_placeholder=None,
                 verbose=False, doors_have_area=True, env_seed=time.time_ns(), **kwargs):
        assert frames_to_stack != 1 and frames_to_stack >= 0, "'frames_to_stack' cannot be negative or 1."
        if kwargs:
            print(f'Following kwargs were passed, but ignored: {kwargs}')

        # Attribute Assignment
        self.env_seed = env_seed
        self.seed(env_seed)
        self._base_rng = np.random.default_rng(self.env_seed)
        if isinstance(movement_properties, dict):
            movement_properties = MovementProperties(**movement_properties)
        self.movement_properties = movement_properties
        self.level_name = level_name
        self._level_shape = None
        self.verbose = verbose
        self.additional_agent_placeholder = additional_agent_placeholder
        self._renderer = None  # expensive - don't use it when not required !
        self._entities = Entities()

        self.n_agents = n_agents

        self.max_steps = max_steps
        self.pomdp_r = pomdp_r
        self.combin_agent_obs = combin_agent_obs
        self.omit_agent_in_obs = omit_agent_in_obs
        self.cast_shadows = cast_shadows
        self.frames_to_stack = frames_to_stack

        self.done_at_collision = done_at_collision
        self.record_episodes = record_episodes
        self.parse_doors = parse_doors
        self.doors_have_area = doors_have_area

        # Reset
        self.reset()

    def __getitem__(self, item):
        return self._entities[item]

    def _base_init_env(self):
        # Objects
        entities = {}
        # Level
        level_filepath = Path(__file__).parent.parent / h.LEVELS_DIR / f'{self.level_name}.txt'
        parsed_level = h.parse_level(level_filepath)
        level_array = h.one_hot_level(parsed_level)
        self._level_shape = level_array.shape

        # Walls
        walls = WallTiles.from_argwhere_coordinates(
            np.argwhere(level_array == c.OCCUPIED_CELL.value),
            self._level_shape
        )
        entities.update({c.WALLS: walls})

        # Floor
        floor = FloorTiles.from_argwhere_coordinates(
            np.argwhere(level_array == c.FREE_CELL.value),
            self._level_shape
        )
        entities.update({c.FLOOR: floor})

        # NOPOS
        self._NO_POS_TILE = Tile(c.NO_POS.value)

        # Doors
        if self.parse_doors:
            parsed_doors = h.one_hot_level(parsed_level, c.DOOR)
            if np.any(parsed_doors):
                door_tiles = [floor.by_pos(pos) for pos in np.argwhere(parsed_doors == c.OCCUPIED_CELL.value)]
                doors = Doors.from_tiles(door_tiles, self._level_shape, context=floor)
                entities.update({c.DOORS: doors})

        # Actions
        self._actions = Actions(self.movement_properties, can_use_doors=self.parse_doors)
        if additional_actions := self.additional_actions:
            self._actions.register_additional_items(additional_actions)

        # Agents
        agents = Agents.from_tiles(floor.empty_tiles[:self.n_agents], self._level_shape,
                                   individual_slices=not self.combin_agent_obs)
        entities.update({c.AGENT: agents})

        if self.additional_agent_placeholder is not None:

            # Empty Observations with either [0, 1, N(0, 1)]
            placeholder = PlaceHolders.from_tiles([self._NO_POS_TILE], self._level_shape,
                                                  fill_value=self.additional_agent_placeholder)

            entities.update({c.AGENT_PLACEHOLDER: placeholder})

        # All entities
        self._entities = Entities()
        self._entities.register_additional_items(entities)

        # Additional Entitites from SubEnvs
        if additional_entities := self.additional_entities:
            self._entities.register_additional_items(additional_entities)

        # Return
        return self._entities

    def _init_obs_cube(self):
        arrays = self._entities.observable_arrays

        # FIXME: Move logic to Register
        if self.omit_agent_in_obs and self.n_agents == 1:
            del arrays[c.AGENT]
        # This does not seem to be necesarry, because this case is allready handled by the Agent Register Class
        # elif self.omit_agent_in_obs:
        #    arrays[c.AGENT] = np.delete(arrays[c.AGENT], 0, axis=0)
        obs_cube_z = sum([a.shape[0] if not self[key].is_per_agent else 1 for key, a in arrays.items()])
        self._obs_cube = np.zeros((obs_cube_z, *self._level_shape), dtype=np.float32)

        # Optionally Pad this obs cube for pomdp cases
        if r := self.pomdp_r:
            x, y = self._level_shape
            # was c.SHADOW
            self._padded_obs_cube = np.full((obs_cube_z, x + r*2, y + r*2), c.SHADOWED_CELL.value, dtype=np.float32)
            self._padded_obs_cube[:, r:r+x, r:r+y] = self._obs_cube

    def reset(self) -> (np.ndarray, int, bool, dict):
        _ = self._base_init_env()
        self._init_obs_cube()
        self.do_additional_reset()

        self._steps = 0

        obs = self._get_observations()
        return obs

    def step(self, actions):

        if self.n_agents == 1 and not isinstance(actions, list):
            actions = [int(actions)]

        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self._steps += 1
        done = False

        # Pre step Hook for later use
        self.hook_pre_step()

        # Move this in a seperate function?
        for action, agent in zip(actions, self[c.AGENT]):
            agent.clear_temp_state()
            action_obj = self._actions[int(action)]
            # self.print(f'Action #{action} has been resolved to: {action_obj}')
            if h.MovingAction.is_member(action_obj):
                valid = self._move_or_colide(agent, action_obj)
            elif h.EnvActions.NOOP == agent.temp_action:
                valid = c.VALID
            elif h.EnvActions.USE_DOOR == action_obj:
                valid = self._handle_door_interaction(agent)
            else:
                valid = self.do_additional_actions(agent, action_obj)
            assert valid is not None, 'This should not happen, every Action musst be detected correctly!'
            agent.temp_action = action_obj
            agent.temp_valid = valid

        # In-between step Hook for later use
        info = self.do_additional_step()

        tiles_with_collisions = self.get_all_tiles_with_collisions()
        for tile in tiles_with_collisions:
            guests = tile.guests_that_can_collide
            for i, guest in enumerate(guests):
                this_collisions = guests[:]
                del this_collisions[i]
                guest.temp_collisions = this_collisions

        if self.done_at_collision and tiles_with_collisions:
            done = True

        # Step the door close intervall
        if self.parse_doors:
            if doors := self[c.DOORS]:
                doors.tick_doors()

        # Finalize
        reward, reward_info = self.calculate_reward()
        info.update(reward_info)
        if self._steps >= self.max_steps:
            done = True
        info.update(step_reward=reward, step=self._steps)
        if self.record_episodes:
            info.update(self._summarize_state())

        # Post step Hook for later use
        info.update(self.hook_post_step())

        obs = self._get_observations()

        return obs, reward, done, info

    def _handle_door_interaction(self, agent) -> c:
        if doors := self[c.DOORS]:
            # Check if agent really is standing on a door:
            if self.doors_have_area:
                door = doors.get_near_position(agent.pos)
            else:
                door = doors.by_pos(agent.pos)
            if door is not None:
                door.use()
                return c.VALID
            # When he doesn't...
            else:
                return c.NOT_VALID
        else:
            return c.NOT_VALID

    def _get_observations(self) -> np.ndarray:
        state_array_dict = self._entities.obs_arrays
        if self.n_agents == 1:
            obs = self._build_per_agent_obs(self[c.AGENT][0], state_array_dict)
        elif self.n_agents >= 2:
            obs = np.stack([self._build_per_agent_obs(agent, state_array_dict) for agent in self[c.AGENT]])
        else:
            raise ValueError('n_agents cannot be smaller than 1!!')
        return obs

    def _build_per_agent_obs(self, agent: Agent, state_array_dict) -> np.ndarray:
        agent_pos_is_omitted = False
        agent_omit_idx = None
        if self.omit_agent_in_obs and self.n_agents == 1:
            # There is only a single agent and we want to omit the agent obs, so just remove the array.
            del state_array_dict[c.AGENT]
        elif self.omit_agent_in_obs and self.combin_agent_obs and self.n_agents > 1:
            state_array_dict[c.AGENT][0, agent.x, agent.y] -= agent.encoding
            agent_pos_is_omitted = True
        elif self.omit_agent_in_obs and not self.combin_agent_obs and self.n_agents > 1:
            agent_omit_idx = next((i for i, a in enumerate(self[c.AGENT]) if a == agent))

        running_idx, shadowing_idxs, can_be_shadowed_idxs = 0, [], []

        for key, array in state_array_dict.items():
            # Flush state array object representation to obs cube
            if not self[key].hide_from_obs_builder:
                if self[key].is_per_agent:
                    per_agent_idx = self[key].idx_by_entity(agent)
                    z = 1
                    self._obs_cube[running_idx: running_idx+z] = array[per_agent_idx]
                else:
                    if key == c.AGENT and agent_omit_idx is not None:
                        z = array.shape[0] - 1
                        for array_idx in range(array.shape[0]):
                            self._obs_cube[running_idx: running_idx+z] = array[[x for x in range(array.shape[0])
                                                                                if x != agent_omit_idx]]
                    elif key == c.AGENT and self.omit_agent_in_obs and self.combin_agent_obs:
                        z = 1
                        self._obs_cube[running_idx: running_idx + z] = array
                    else:
                        z = array.shape[0]
                        self._obs_cube[running_idx: running_idx+z] = array
                # Define which OBS SLices cast a Shadow
                if self[key].is_blocking_light:
                    for i in range(z):
                        shadowing_idxs.append(running_idx + i)
                # Define which OBS SLices are effected by shadows
                if self[key].can_be_shadowed:
                    for i in range(z):
                        can_be_shadowed_idxs.append(running_idx + i)
                running_idx += z

        if agent_pos_is_omitted:
            state_array_dict[c.AGENT][0, agent.x, agent.y] += agent.encoding

        if r := self.pomdp_r:
            self._padded_obs_cube[:] = c.SHADOWED_CELL.value   # Was c.SHADOW
            # self._padded_obs_cube[0] = c.OCCUPIED_CELL.value
            x, y = self._level_shape
            self._padded_obs_cube[:, r:r + x, r:r + y] = self._obs_cube
            global_x, global_y = map(sum, zip(agent.pos, (r, r)))
            x0, x1 = max(0, global_x - self.pomdp_r), global_x + self.pomdp_r + 1
            y0, y1 = max(0, global_y - self.pomdp_r), global_y + self.pomdp_r + 1
            obs = self._padded_obs_cube[:, x0:x1, y0:y1]
        else:
            obs = self._obs_cube

        if self.cast_shadows:
            obs_block_light = [obs[idx] != c.OCCUPIED_CELL.value for idx in shadowing_idxs]
            door_shadowing = False
            if self.parse_doors:
                if doors := self[c.DOORS]:
                    if door := doors.by_pos(agent.pos):
                        if door.is_closed:
                            for group in door.connectivity_subgroups:
                                if agent.last_pos not in group:
                                    door_shadowing = True
                                    if self.pomdp_r:
                                        blocking = [tuple(np.subtract(x, agent.pos) + (self.pomdp_r, self.pomdp_r))
                                                    for x in group]
                                        xs, ys = zip(*blocking)
                                    else:
                                        xs, ys = zip(*group)

                                    # noinspection PyUnresolvedReferences
                                    obs_block_light[0][xs, ys] = False

            light_block_map = Map((np.prod(obs_block_light, axis=0) != True).astype(int))
            if self.pomdp_r:
                light_block_map = light_block_map.do_fov(self.pomdp_r, self.pomdp_r, max(self._level_shape))
            else:
                light_block_map = light_block_map.do_fov(*agent.pos, max(self._level_shape))
            if door_shadowing:
                # noinspection PyUnboundLocalVariable
                light_block_map[xs, ys] = 0
            agent.temp_light_map = light_block_map
            for obs_idx in can_be_shadowed_idxs:
                obs[obs_idx] = ((obs[obs_idx] * light_block_map) + 0.) - (1 - light_block_map)  # * obs[0])
        else:
            pass

        # Additional Observation:
        for additional_obs in self.additional_obs_build():
            obs[running_idx:running_idx+additional_obs.shape[0]] = additional_obs
            running_idx += additional_obs.shape[0]
        for additional_per_agent_obs in self.additional_per_agent_obs_build(agent):
            obs[running_idx:running_idx + additional_per_agent_obs.shape[0]] = additional_per_agent_obs
            running_idx += additional_per_agent_obs.shape[0]

        return obs

    def get_all_tiles_with_collisions(self) -> List[Tile]:
        tiles_with_collisions = list()
        for tile in self[c.FLOOR]:
            if tile.is_occupied():
                guests = [guest for guest in tile.guests if guest.can_collide]
                if len(guests) >= 2:
                    tiles_with_collisions.append(tile)
        return tiles_with_collisions

    def _move_or_colide(self, agent: Agent, action: Action) -> Constants:
        new_tile, valid = self._check_agent_move(agent, action)
        if valid:
            # Does not collide width level boundaries
            return agent.move(new_tile)
        else:
            # Agent seems to be trying to collide in this step
            return c.NOT_VALID

    def _check_agent_move(self, agent, action: Action) -> (Tile, bool):
        # Actions
        x_diff, y_diff = h.ACTIONMAP[action.identifier]
        x_new = agent.x + x_diff
        y_new = agent.y + y_diff

        new_tile = self[c.FLOOR].by_pos((x_new, y_new))
        if new_tile:
            valid = c.VALID
        else:
            tile = agent.tile
            valid = c.VALID
            return tile, valid

        if self.parse_doors and agent.last_pos != c.NO_POS:
            if doors := self[c.DOORS]:
                if self.doors_have_area:
                    if door := doors.by_pos(new_tile.pos):
                        if door.can_collide:
                            return agent.tile, c.NOT_VALID
                        else:  # door.is_closed:
                            pass

                if door := doors.by_pos(agent.pos):
                    if door.is_open:
                        pass
                    else:  # door.is_closed:
                        if door.is_linked(agent.last_pos, new_tile.pos):
                            pass
                        else:
                            return agent.tile, c.NOT_VALID
                else:
                    pass
        else:
            pass

        return new_tile, valid

    def calculate_reward(self) -> (int, dict):
        # Returns: Reward, Info
        info_dict = dict()
        reward = 0

        for agent in self[c.AGENT]:
            if self._actions.is_moving_action(agent.temp_action):
                if agent.temp_valid:
                    # info_dict.update(movement=1)
                    # info_dict.update({f'{agent.name}_failed_action': 1})
                    # reward += 0.00
                    pass
                else:
                    # self.print('collision')
                    reward -= 0.01
                    self.print(f'{agent.name} just hit the wall at {agent.pos}.')
                    info_dict.update({f'{agent.name}_vs_LEVEL': 1})

            elif h.EnvActions.USE_DOOR == agent.temp_action:
                if agent.temp_valid:
                    # reward += 0.00
                    self.print(f'{agent.name} did just use the door at {agent.pos}.')
                    info_dict.update(door_used=1)
                else:
                    # reward -= 0.00
                    self.print(f'{agent.name} just tried to use a door at {agent.pos}, but failed.')
                    info_dict.update({f'{agent.name}_failed_action': 1})
                    info_dict.update({f'{agent.name}_failed_door_open': 1})
            elif h.EnvActions.NOOP == agent.temp_action:
                info_dict.update(no_op=1)
                # reward -= 0.00

            additional_reward, additional_info_dict = self.calculate_additional_reward(agent)
            reward += additional_reward
            info_dict.update(additional_info_dict)

            if agent.temp_collisions:
                self.print(f't = {self._steps}\t{agent.name} has collisions with {agent.temp_collisions}')

            for other_agent in agent.temp_collisions:
                info_dict.update({f'{agent.name}_vs_{other_agent.name}': 1})

        self.print(f"reward is {reward}")
        return reward, info_dict

    def render(self, mode='human'):
        if not self._renderer:  # lazy init
            height, width = self._obs_cube.shape[1:]
            self._renderer = Renderer(width, height, view_radius=self.pomdp_r, fps=5)

        walls = [RenderEntity('wall', wall.pos) for wall in self[c.WALLS]]

        agents = []
        for i, agent in enumerate(self[c.AGENT]):
            name, state = h.asset_str(agent)
            agents.append(RenderEntity(name, agent.pos, 1, 'none', state, i + 1, agent.temp_light_map))
        doors = []
        if self.parse_doors:
            for i, door in enumerate(self[c.DOORS]):
                name, state = 'door_open' if door.is_open else 'door_closed', 'blank'
                doors.append(RenderEntity(name, door.pos, 1, 'none', state, i + 1))
        additional_assets = self.render_additional_assets()

        self._renderer.render(walls + doors + additional_assets + agents)

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        # d = {key: val._asdict() if hasattr(val, '_asdict') else val for key, val in self.__dict__.items()
        d = {key: val for key, val in self.__dict__.items() if not key.startswith('_') and not key.startswith('__')}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w') as f:
            simplejson.dump(d, f, indent=4, namedtuple_as_object=True)

    def _summarize_state(self):
        summary = {f'{REC_TAC}step': self._steps}

        for entity_group in self._entities:
            summary.update({f'{REC_TAC}{entity_group.name}': entity_group.summarize_states(n_steps=self._steps)})
        return summary

    def print(self, string):
        if self.verbose:
            print(string)

    # Properties which are called by the base class to extend beyond attributes of the base class
    @property
    def additional_actions(self) -> Union[Action, List[Action]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A list of Actions-object holding all additional actions.
        :rtype:             List[Action]
        """
        return []

    @property
    def additional_entities(self) -> Dict[(Enum, Entities)]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A single Entites collection or a list of such.
        :rtype:             Union[Entities, List[Entities]]
        """
        return {}

    # Functions which provide additions to functions of the base class
    #  Always call super!!!!!!
    @abc.abstractmethod
    def additional_obs_build(self) -> List[np.ndarray]:
        return []

    @abc.abstractmethod
    def additional_per_agent_obs_build(self, agent) -> List[np.ndarray]:
        return []

    @abc.abstractmethod
    def do_additional_reset(self) -> None:
        pass

    @abc.abstractmethod
    def do_additional_step(self) -> dict:
        return {}

    @abc.abstractmethod
    def do_additional_actions(self, agent: Agent, action: Action) -> Union[None, c]:
        return None

    @abc.abstractmethod
    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        return 0, {}

    @abc.abstractmethod
    def render_additional_assets(self):
        return []

    # Hooks for in between operations.
    #  Always call super!!!!!!
    @abc.abstractmethod
    def hook_pre_step(self) -> None:
        pass

    @abc.abstractmethod
    def hook_post_step(self) -> dict:
        return {}
