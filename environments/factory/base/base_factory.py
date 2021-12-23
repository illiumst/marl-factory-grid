import abc
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Union, Iterable, Dict
import numpy as np

import gym
from gym import spaces
from gym.wrappers import FrameStack

from environments.factory.base.shadow_casting import Map
from environments import helpers as h
from environments.helpers import Constants as c
from environments.factory.base.objects import Agent, Tile, Action
from environments.factory.base.registers import Actions, Entities, Agents, Doors, FloorTiles, WallTiles, PlaceHolders, \
    GlobalPositions
from environments.utility_classes import MovementProperties, ObservationProperties, MarlFrameStack
from environments.utility_classes import AgentRenderOptions as a_obs

import simplejson

REC_TAC = 'rec_'


# noinspection PyAttributeOutsideInit
class BaseFactory(gym.Env):

    @property
    def action_space(self):
        return spaces.Discrete(len(self._actions))

    @property
    def named_action_space(self):
        return {x.identifier.value: idx for idx, x in enumerate(self._actions.values())}

    @property
    def observation_space(self):
        obs, _ = self._build_observations()
        if self.n_agents > 1:
            shape = obs[0].shape
        else:
            shape = obs.shape
        space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        return space

    @property
    def named_observation_space(self):
        # Build it
        _, named_obs = self._build_observations()
        if self.n_agents > 1:
            # Only return the first named obs space, as their structure at the moment is same.
            return named_obs[list(named_obs.keys())[0]]
        else:
            return named_obs

    @property
    def pomdp_diameter(self):
        return self._pomdp_r * 2 + 1

    @property
    def movement_actions(self):
        return self._actions.movement_actions

    @property
    def params(self) -> dict:
        d = {key: val for key, val in self.__dict__.items() if not key.startswith('_') and not key.startswith('__')}
        return d

    def __enter__(self):
        return self if self.obs_prop.frames_to_stack == 0 else \
            MarlFrameStack(FrameStack(self, self.obs_prop.frames_to_stack))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, level_name='simple', n_agents=1, max_steps=int(5e2),
                 mv_prop: MovementProperties = MovementProperties(),
                 obs_prop: ObservationProperties = ObservationProperties(),
                 parse_doors=False, done_at_collision=False, inject_agents: Union[None, List] = None,
                 verbose=False, doors_have_area=True, env_seed=time.time_ns(), individual_rewards=False,
                 **kwargs):

        if isinstance(mv_prop, dict):
            mv_prop = MovementProperties(**mv_prop)
        if isinstance(obs_prop, dict):
            obs_prop = ObservationProperties(**obs_prop)

        assert obs_prop.frames_to_stack != 1 and \
               obs_prop.frames_to_stack >= 0, "'frames_to_stack' cannot be negative or 1."
        if kwargs:
            print(f'Following kwargs were passed, but ignored: {kwargs}')

        # Attribute Assignment
        self.env_seed = env_seed
        self.seed(env_seed)
        self._base_rng = np.random.default_rng(self.env_seed)
        self.mv_prop = mv_prop
        self.obs_prop = obs_prop
        self.level_name = level_name
        self._level_shape = None
        self._obs_shape = None
        self.verbose = verbose
        self._renderer = None  # expensive - don't use it when not required !
        self._entities = Entities()

        self.n_agents = n_agents
        level_filepath = Path(__file__).parent.parent / h.LEVELS_DIR / f'{self.level_name}.txt'
        self._parsed_level = h.parse_level(level_filepath)

        self.max_steps = max_steps
        self._pomdp_r = self.obs_prop.pomdp_r

        self.done_at_collision = done_at_collision
        self._record_episodes = False
        self.parse_doors = parse_doors
        self._injected_agents = inject_agents or []
        self.doors_have_area = doors_have_area
        self.individual_rewards = individual_rewards

        # Reset
        self.reset()

    def __getitem__(self, item):
        return self._entities[item]

    def _base_init_env(self):

        # All entities
        # Objects
        self._entities = Entities()
        # Level

        level_array = h.one_hot_level(self._parsed_level)
        level_array = np.pad(level_array, self.obs_prop.pomdp_r, 'constant', constant_values=1)

        self._level_shape = level_array.shape
        self._obs_shape = self._level_shape if not self.obs_prop.pomdp_r else (self.pomdp_diameter, ) * 2

        # Walls
        walls = WallTiles.from_argwhere_coordinates(
            np.argwhere(level_array == c.OCCUPIED_CELL),
            self._level_shape
        )
        self._entities.register_additional_items({c.WALLS: walls})

        # Floor
        floor = FloorTiles.from_argwhere_coordinates(
            np.argwhere(level_array == c.FREE_CELL),
            self._level_shape
        )
        self._entities.register_additional_items({c.FLOOR: floor})

        # NOPOS
        self._NO_POS_TILE = Tile(c.NO_POS, None)

        # Doors
        if self.parse_doors:
            parsed_doors = h.one_hot_level(self._parsed_level, c.DOOR)
            parsed_doors = np.pad(parsed_doors, self.obs_prop.pomdp_r, 'constant', constant_values=0)
            if np.any(parsed_doors):
                door_tiles = [floor.by_pos(tuple(pos)) for pos in np.argwhere(parsed_doors == c.OCCUPIED_CELL)]
                doors = Doors.from_tiles(door_tiles, self._level_shape,
                                         entity_kwargs=dict(context=floor)
                                         )
                self._entities.register_additional_items({c.DOORS: doors})

        # Actions
        self._actions = Actions(self.mv_prop, can_use_doors=self.parse_doors)
        if additional_actions := self.additional_actions:
            self._actions.register_additional_items(additional_actions)

        # Agents
        agents_to_spawn = self.n_agents-len(self._injected_agents)
        agents_kwargs = dict(individual_slices=self.obs_prop.render_agents == a_obs.SEPERATE,
                             hide_from_obs_builder=self.obs_prop.render_agents in [a_obs.NOT, a_obs.LEVEL],
                             )
        if agents_to_spawn:
            agents = Agents.from_tiles(floor.empty_tiles[:agents_to_spawn], self._level_shape, **agents_kwargs)
        else:
            agents = Agents(**agents_kwargs)
        if self._injected_agents:
            initialized_injections = list()
            for i, injection in enumerate(self._injected_agents):
                agents.register_item(injection(self, floor.empty_tiles[agents_to_spawn+i+1], static_problem=False))
                initialized_injections.append(agents[-1])
            self._initialized_injections = initialized_injections
        self._entities.register_additional_items({c.AGENT: agents})

        if self.obs_prop.additional_agent_placeholder is not None:
            # TODO: Make this accept Lists for multiple placeholders

            # Empty Observations with either [0, 1, N(0, 1)]
            placeholder = PlaceHolders.from_values(self.obs_prop.additional_agent_placeholder, self._level_shape,
                                                   entity_kwargs=dict(
                                                       fill_value=self.obs_prop.additional_agent_placeholder)
                                                   )

            self._entities.register_additional_items({c.AGENT_PLACEHOLDER: placeholder})

        # Additional Entitites from SubEnvs
        if additional_entities := self.additional_entities:
            self._entities.register_additional_items(additional_entities)

        if self.obs_prop.show_global_position_info:
            global_positions = GlobalPositions(self._level_shape)
            obs_shape_2d = self._level_shape if not self._pomdp_r else ((self.pomdp_diameter,) * 2)
            global_positions.spawn_global_position_objects(obs_shape_2d, self[c.AGENT])
            self._entities.register_additional_items({c.GLOBAL_POSITION: global_positions})

        # Return
        return self._entities

    def reset(self) -> (np.typing.ArrayLike, int, bool, dict):
        _ = self._base_init_env()
        self.do_additional_reset()

        self._steps = 0

        obs, _ = self._build_observations()
        return obs

    def step(self, actions):

        if self.n_agents == 1 and not isinstance(actions, list):
            actions = [int(actions)]

        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self._steps += 1

        # Pre step Hook for later use
        self.hook_pre_step()

        # Move this in a seperate function?
        for action, agent in zip(actions, self[c.AGENT]):
            agent.clear_temp_state()
            action_obj = self._actions[int(action)]
            # cls.print(f'Action #{action} has been resolved to: {action_obj}')
            if h.EnvActions.is_move(action_obj):
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

        done = self.done_at_collision and tiles_with_collisions

        done = done or self.check_additional_done()

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
        if self._record_episodes:
            info.update(self._summarize_state())

        # Post step Hook for later use
        info.update(self.hook_post_step())

        obs, _ = self._build_observations()

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

    def _build_observations(self) -> np.typing.ArrayLike:
        # Observation dict:
        per_agent_expl_idx = dict()
        per_agent_obsn = dict()
        # Generel Observations
        lvl_obs = self[c.WALLS].as_array()
        door_obs = self[c.DOORS].as_array()
        agent_obs = self[c.AGENT].as_array() if self.obs_prop.render_agents != a_obs.NOT else None
        placeholder_obs = self[c.AGENT_PLACEHOLDER].as_array() if self[c.AGENT_PLACEHOLDER] else None
        add_obs_dict = self._additional_observations()

        for agent_idx, agent in enumerate(self[c.AGENT]):
            obs_dict = dict()
            # Build Agent Observations
            if self.obs_prop.render_agents != a_obs.NOT:
                if self.obs_prop.omit_agent_self:
                    if self.obs_prop.render_agents == a_obs.SEPERATE:
                        agent_obs = np.take(agent_obs, [x for x in range(self.n_agents) if x != agent_idx], axis=0)
                    else:
                        agent_obs = agent_obs.copy()
                        agent_obs[(0, *agent.pos)] -= agent.encoding

            # Build Level Observations
            if self.obs_prop.render_agents == a_obs.LEVEL:
                lvl_obs = lvl_obs.copy()
                lvl_obs += agent_obs

            obs_dict[c.WALLS] = lvl_obs
            if self.obs_prop.render_agents in [a_obs.SEPERATE, a_obs.COMBINED]:
                obs_dict[c.AGENT] = agent_obs
            if self[c.AGENT_PLACEHOLDER]:
                obs_dict[c.AGENT_PLACEHOLDER] = placeholder_obs
            obs_dict[c.DOORS] = door_obs
            obs_dict.update(add_obs_dict)
            obsn = np.vstack(list(obs_dict.values()))
            if self.obs_prop.pomdp_r:
                obsn = self._do_pomdp_cutout(agent, obsn)

            raw_obs = self._additional_per_agent_raw_observations(agent)
            obsn = np.vstack((obsn, *list(raw_obs.values())))

            keys = list(chain(obs_dict.keys(), raw_obs.keys()))
            idxs = np.cumsum([x.shape[0] for x in chain(obs_dict.values(), raw_obs.values())]) - 1
            per_agent_expl_idx[agent.name] = {key: list(range(a, b)) for key, a, b in
                                              zip(keys, idxs, list(idxs[1:]) + [idxs[-1]+1, ])}

            # Shadow Casting
            try:
                light_block_obs = [obs_idx for key, obs_idx in per_agent_expl_idx[agent.name].items()
                                   if self[key].is_blocking_light]
                # Flatten
                light_block_obs = [x for y in light_block_obs for x in y]
                shadowed_obs = [obs_idx for key, obs_idx in per_agent_expl_idx[agent.name].items()
                                if self[key].can_be_shadowed]
                # Flatten
                shadowed_obs = [x for y in shadowed_obs for x in y]
            except AttributeError as e:
                print('Check your Keys! Only use Constants as Keys!')
                print(e)
                raise e
            if self.obs_prop.cast_shadows:
                obs_block_light = obsn[light_block_obs] != c.OCCUPIED_CELL
                door_shadowing = False
                if self.parse_doors:
                    if doors := self[c.DOORS]:
                        if door := doors.by_pos(agent.pos):
                            if door.is_closed:
                                for group in door.connectivity_subgroups:
                                    if agent.last_pos not in group:
                                        door_shadowing = True
                                        if self._pomdp_r:
                                            blocking = [
                                                tuple(np.subtract(x, agent.pos) + (self._pomdp_r, self._pomdp_r))
                                                for x in group]
                                            xs, ys = zip(*blocking)
                                        else:
                                            xs, ys = zip(*group)

                                        # noinspection PyUnresolvedReferences
                                        obs_block_light[:, xs, ys] = False

                light_block_map = Map((np.prod(obs_block_light, axis=0) != True).astype(int).squeeze())
                if self._pomdp_r:
                    light_block_map = light_block_map.do_fov(self._pomdp_r, self._pomdp_r, max(self._level_shape))
                else:
                    light_block_map = light_block_map.do_fov(*agent.pos, max(self._level_shape))
                if door_shadowing:
                    # noinspection PyUnboundLocalVariable
                    light_block_map[xs, ys] = 0
                agent.temp_light_map = light_block_map.copy()

                obsn[shadowed_obs] = ((obsn[shadowed_obs] * light_block_map) + 0.) - (1 - light_block_map)
            else:
                pass

            per_agent_obsn[agent.name] = obsn

        if self.n_agents == 1:
            agent_name = self[c.AGENT][0].name
            obs, explained_idx = per_agent_obsn[agent_name], per_agent_expl_idx[agent_name]
        elif self.n_agents >= 2:
            obs, explained_idx = np.stack(list(per_agent_obsn.values())), per_agent_expl_idx
        else:
            raise ValueError

        return obs, explained_idx

    def _do_pomdp_cutout(self, agent, obs_to_be_padded):
        assert obs_to_be_padded.ndim == 3
        r, d = self._pomdp_r, self.pomdp_diameter
        x0, x1 = max(0, agent.x - r), min(agent.x + r + 1, self._level_shape[0])
        y0, y1 = max(0, agent.y - r), min(agent.y + r + 1, self._level_shape[1])
        oobs = obs_to_be_padded[:, x0:x1, y0:y1]
        if oobs.shape[1:] != (d, d):
            if xd := oobs.shape[1] % d:
                if agent.x > r:
                    x0_pad = 0
                    x1_pad = (d - xd)
                else:
                    x0_pad = r - agent.x
                    x1_pad = 0
            else:
                x0_pad, x1_pad = 0, 0

            if yd := oobs.shape[2] % d:
                if agent.y > r:
                    y0_pad = 0
                    y1_pad = (d - yd)
                else:
                    y0_pad = r - agent.y
                    y1_pad = 0
            else:
                y0_pad, y1_pad = 0, 0

            oobs = np.pad(oobs, ((0, 0), (x0_pad, x1_pad), (y0_pad, y1_pad)), 'constant')
        return oobs

    def get_all_tiles_with_collisions(self) -> List[Tile]:
        tiles_with_collisions = list()
        for tile in self[c.FLOOR]:
            if tile.is_occupied():
                guests = tile.guests_that_can_collide
                if len(guests) >= 2:
                    tiles_with_collisions.append(tile)
        return tiles_with_collisions

    def _move_or_colide(self, agent: Agent, action: Action) -> bool:
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
                        if door.is_open:
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
        per_agent_info_dict = defaultdict(dict)
        reward = {}

        for agent in self[c.AGENT]:
            per_agent_reward = 0
            if self._actions.is_moving_action(agent.temp_action):
                if agent.temp_valid:
                    # info_dict.update(movement=1)
                    per_agent_reward -= 0.001
                    pass
                else:
                    per_agent_reward -= 0.05
                    self.print(f'{agent.name} just hit the wall at {agent.pos}.')
                    per_agent_info_dict[agent.name].update({f'{agent.name}_vs_LEVEL': 1})

            elif h.EnvActions.USE_DOOR == agent.temp_action:
                if agent.temp_valid:
                    # per_agent_reward += 0.00
                    self.print(f'{agent.name} did just use the door at {agent.pos}.')
                    per_agent_info_dict[agent.name].update(door_used=1)
                else:
                    # per_agent_reward -= 0.00
                    self.print(f'{agent.name} just tried to use a door at {agent.pos}, but failed.')
                    per_agent_info_dict[agent.name].update({f'{agent.name}_failed_door_open': 1})
            elif h.EnvActions.NOOP == agent.temp_action:
                per_agent_info_dict[agent.name].update(no_op=1)
                # per_agent_reward -= 0.00

            # EnvMonitor Notes
            if agent.temp_valid:
                per_agent_info_dict[agent.name].update(valid_action=1)
                per_agent_info_dict[agent.name].update({f'{agent.name}_valid_action': 1})
            else:
                per_agent_info_dict[agent.name].update(failed_action=1)
                per_agent_info_dict[agent.name].update({f'{agent.name}_failed_action': 1})

            additional_reward, additional_info_dict = self.calculate_additional_reward(agent)
            per_agent_reward += additional_reward
            per_agent_info_dict[agent.name].update(additional_info_dict)

            if agent.temp_collisions:
                self.print(f't = {self._steps}\t{agent.name} has collisions with {agent.temp_collisions}')
                per_agent_info_dict[agent.name].update(collisions=1)

                for other_agent in agent.temp_collisions:
                    per_agent_info_dict[agent.name].update({f'{agent.name}_vs_{other_agent.name}': 1})
            reward[agent.name] = per_agent_reward

        # Combine the per_agent_info_dict:
        combined_info_dict = defaultdict(lambda: 0)
        for info_dict in per_agent_info_dict.values():
            for key, value in info_dict.items():
                combined_info_dict[key] += value
        combined_info_dict = dict(combined_info_dict)

        if self.individual_rewards:
            self.print(f"rewards are {reward}")
            reward = list(reward.values())
            return reward, combined_info_dict
        else:
            reward = sum(reward.values())
            self.print(f"reward is {reward}")
        return reward, combined_info_dict

    # noinspection PyGlobalUndefined
    def render(self, mode='human'):
        if not self._renderer:  # lazy init
            from environments.factory.base.renderer import Renderer, RenderEntity
            global Renderer, RenderEntity
            height, width = self._level_shape
            self._renderer = Renderer(width, height, view_radius=self._pomdp_r, fps=5)

        # noinspection PyUnboundLocalVariable
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

        return self._renderer.render(walls + doors + additional_assets + agents)

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        d = self.params
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w') as f:
            simplejson.dump(d, f, indent=4, namedtuple_as_object=True)

    def get_injected_agents(self) -> list:
        if hasattr(self, '_initialized_injections'):
            return self._initialized_injections
        else:
            return []

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
    def additional_entities(self) -> Dict[(str, Entities)]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A single Entites collection or a list of such.
        :rtype:             Union[Entities, List[Entities]]
        """
        return {}

    # Functions which provide additions to functions of the base class
    #  Always call super!!!!!!
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
    def check_additional_done(self) -> bool:
        return False

    @abc.abstractmethod
    def _additional_observations(self) -> Dict[str, np.typing.ArrayLike]:
        return {}

    @abc.abstractmethod
    def _additional_per_agent_raw_observations(self, agent) -> Dict[str, np.typing.ArrayLike]:
        additional_raw_observations = {}
        if self.obs_prop.show_global_position_info:
            additional_raw_observations.update({c.GLOBAL_POSITION: self[c.GLOBAL_POSITION].by_entity(agent).as_array()})
        return additional_raw_observations

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
