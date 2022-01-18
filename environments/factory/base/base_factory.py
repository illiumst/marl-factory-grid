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
from environments.helpers import EnvActions as a
from environments.helpers import RewardsBase
from environments.factory.base.objects import Agent, Floor, Action
from environments.factory.base.registers import Actions, Entities, Agents, Doors, Floors, Walls, PlaceHolders, \
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
        return {x.identifier: idx for idx, x in enumerate(self._actions.values())}

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
        d['class_name'] = self.__class__.__name__
        return d

    def __enter__(self):
        return self if self.obs_prop.frames_to_stack == 0 else \
            MarlFrameStack(FrameStack(self, self.obs_prop.frames_to_stack))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, level_name='simple', n_agents=1, max_steps=int(5e2),
                 mv_prop: MovementProperties = MovementProperties(),
                 obs_prop: ObservationProperties = ObservationProperties(),
                 rewards_base: RewardsBase = RewardsBase(),
                 parse_doors=False, done_at_collision=False, inject_agents: Union[None, List] = None,
                 verbose=False, doors_have_area=True, env_seed=time.time_ns(), individual_rewards=False,
                 class_name='', **kwargs):

        if class_name:
            print(f'You loaded parameters for {class_name}', f'this is: {self.__class__.__name__}')

        if isinstance(mv_prop, dict):
            mv_prop = MovementProperties(**mv_prop)
        if isinstance(obs_prop, dict):
            obs_prop = ObservationProperties(**obs_prop)
        if isinstance(rewards_base, dict):
            rewards_base = RewardsBase(**rewards_base)

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
        self.rewards_base = rewards_base
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

        # TODO: Reset ---> document this
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
        walls = Walls.from_argwhere_coordinates(
            np.argwhere(level_array == c.OCCUPIED_CELL),
            self._level_shape
        )
        self._entities.register_additional_items({c.WALLS: walls})

        # Floor
        floor = Floors.from_argwhere_coordinates(
            np.argwhere(level_array == c.FREE_CELL),
            self._level_shape
        )
        self._entities.register_additional_items({c.FLOOR: floor})

        # NOPOS
        self._NO_POS_TILE = Floor(c.NO_POS, None)

        # Doors
        if self.parse_doors:
            parsed_doors = h.one_hot_level(self._parsed_level, c.DOOR)
            parsed_doors = np.pad(parsed_doors, self.obs_prop.pomdp_r, 'constant', constant_values=0)
            if np.any(parsed_doors):
                door_tiles = [floor.by_pos(tuple(pos)) for pos in np.argwhere(parsed_doors == c.OCCUPIED_CELL)]
                doors = Doors.from_tiles(door_tiles, self._level_shape, have_area=self.doors_have_area,
                                         entity_kwargs=dict(context=floor)
                                         )
                self._entities.register_additional_items({c.DOORS: doors})

        # Actions
        self._actions = Actions(self.mv_prop, can_use_doors=self.parse_doors)
        if additional_actions := self.actions_hook:
            self._actions.register_additional_items(additional_actions)

        # Agents
        agents_to_spawn = self.n_agents-len(self._injected_agents)
        agents_kwargs = dict(individual_slices=self.obs_prop.render_agents == a_obs.SEPERATE,
                             hide_from_obs_builder=self.obs_prop.render_agents in [a_obs.NOT, a_obs.LEVEL],
                             )
        if agents_to_spawn:
            agents = Agents.from_tiles(floor.empty_tiles[:agents_to_spawn], self._level_shape, **agents_kwargs)
        else:
            agents = Agents(self._level_shape, **agents_kwargs)
        if self._injected_agents:
            initialized_injections = list()
            for i, injection in enumerate(self._injected_agents):
                agents.register_item(injection(self, floor.empty_tiles[0], agents, static_problem=False))
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
        if additional_entities := self.entities_hook:
            self._entities.register_additional_items(additional_entities)

        if self.obs_prop.show_global_position_info:
            global_positions = GlobalPositions(self._level_shape)
            # This moved into the GlobalPosition object
            # obs_shape_2d = self._level_shape if not self._pomdp_r else ((self.pomdp_diameter,) * 2)
            global_positions.spawn_global_position_objects(self[c.AGENT])
            self._entities.register_additional_items({c.GLOBAL_POSITION: global_positions})

        # Return
        return self._entities

    def reset(self) -> (np.typing.ArrayLike, int, bool, dict):
        _ = self._base_init_env()
        self.reset_hook()

        self._steps = 0

        obs, _ = self._build_observations()
        return obs

    def step(self, actions):

        if self.n_agents == 1 and not isinstance(actions, list):
            actions = [int(actions)]

        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self._steps += 1

        # Pre step Hook for later use
        self.pre_step_hook()

        for action, agent in zip(actions, self[c.AGENT]):
            agent.clear_temp_state()
            action_obj = self._actions[int(action)]
            step_result = dict(collisions=[], rewards=[], info={}, action_name='', action_valid=False)
            # cls.print(f'Action #{action} has been resolved to: {action_obj}')
            if a.is_move(action_obj):
                action_valid, reward = self._do_move_action(agent, action_obj)
            elif a.NOOP == action_obj:
                action_valid = c.VALID
                reward = dict(value=self.rewards_base.NOOP, reason=a.NOOP, info={f'{agent.name}_NOOP': 1, 'NOOP': 1})
            elif a.USE_DOOR == action_obj:
                action_valid, reward = self._handle_door_interaction(agent)
            else:
                # noinspection PyTupleAssignmentBalance
                action_valid, reward = self.do_additional_actions(agent, action_obj)
                # Not needed any more sice the tuple assignment above will fail in case of a failing action resolvement.
                # assert step_result is not None, 'This should not happen, every Action musst be detected correctly!'
            step_result['action_name'] = action_obj.identifier
            step_result['action_valid'] = action_valid
            step_result['rewards'].append(reward)
            agent.step_result = step_result

        # Additional step and Reward, Info Init
        rewards, info = self.step_hook()
        # Todo: Make this faster, so that only tiles of entities that can collide are searched.
        tiles_with_collisions = self.get_all_tiles_with_collisions()
        for tile in tiles_with_collisions:
            guests = tile.guests_that_can_collide
            for i, guest in enumerate(guests):
                # This does make a copy, but is faster than.copy()
                this_collisions = guests[:]
                del this_collisions[i]
                assert hasattr(guest, 'step_result')
                for collision in this_collisions:
                    guest.step_result['collisions'].append(collision)

        done = False
        if self.done_at_collision:
            if done_at_col := bool(tiles_with_collisions):
                done = done_at_col
                info.update(COLLISION_DONE=done_at_col)

        additional_done, additional_done_info = self.check_additional_done()
        done = done or additional_done
        info.update(additional_done_info)

        # Step the door close intervall
        if self.parse_doors:
            if doors := self[c.DOORS]:
                doors.tick_doors()

        # Finalize
        reward, reward_info = self.build_reward_result(rewards)

        info.update(reward_info)
        if self._steps >= self.max_steps:
            done = True
        info.update(step_reward=reward, step=self._steps)
        if self._record_episodes:
            info.update(self._summarize_state())

        # Post step Hook for later use
        info.update(self.post_step_hook())

        obs, _ = self._build_observations()

        return obs, reward, done, info

    def _handle_door_interaction(self, agent) -> (bool, dict):
        if doors := self[c.DOORS]:
            # Check if agent really is standing on a door:
            if self.doors_have_area:
                door = doors.get_near_position(agent.pos)
            else:
                door = doors.by_pos(agent.pos)
            if door is not None:
                door.use()
                valid = c.VALID
                self.print(f'{agent.name} just used a {door.name} at {door.pos}')
                info_dict = {f'{agent.name}_door_use': 1, f'door_use': 1}
            # When he doesn't...
            else:
                valid = c.NOT_VALID
                info_dict = {f'{agent.name}_failed_door_use': 1, 'failed_door_use': 1}
                self.print(f'{agent.name} just tried to use a door at {agent.pos}, but there is none.')

        else:
            raise RuntimeError('This should not happen, since the door action should not be available.')
        reward = dict(value=self.rewards_base.USE_DOOR_VALID if valid else self.rewards_base.USE_DOOR_FAIL,
                      reason=a.USE_DOOR, info=info_dict)

        return valid, reward

    def _build_observations(self) -> np.typing.ArrayLike:
        # Observation dict:
        per_agent_expl_idx = dict()
        per_agent_obsn = dict()
        # Generel Observations
        lvl_obs = self[c.WALLS].as_array()
        door_obs = self[c.DOORS].as_array() if self.parse_doors else None
        if self.obs_prop.render_agents == a_obs.NOT:
            global_agent_obs = None
        elif self.obs_prop.omit_agent_self and self.n_agents == 1:
            global_agent_obs = None
        else:
            global_agent_obs = self[c.AGENT].as_array().copy()
        placeholder_obs = self[c.AGENT_PLACEHOLDER].as_array() if self[c.AGENT_PLACEHOLDER] else None
        add_obs_dict = self.observations_hook()

        for agent_idx, agent in enumerate(self[c.AGENT]):
            obs_dict = dict()
            # Build Agent Observations
            if self.obs_prop.render_agents != a_obs.NOT:
                if self.obs_prop.omit_agent_self and self.n_agents >= 2:
                    if self.obs_prop.render_agents == a_obs.SEPERATE:
                        other_agent_obs_idx = [x for x in range(self.n_agents) if x != agent_idx]
                        agent_obs = np.take(global_agent_obs, other_agent_obs_idx, axis=0)
                    else:
                        agent_obs = global_agent_obs.copy()
                        agent_obs[(0, *agent.pos)] -= agent.encoding
                else:
                    agent_obs = global_agent_obs
            else:
                agent_obs = global_agent_obs

            # Build Level Observations
            if self.obs_prop.render_agents == a_obs.LEVEL:
                lvl_obs = lvl_obs.copy()
                lvl_obs += global_agent_obs

            obs_dict[c.WALLS] = lvl_obs
            if self.obs_prop.render_agents in [a_obs.SEPERATE, a_obs.COMBINED] and agent_obs is not None:
                obs_dict[c.AGENT] = agent_obs[:]
            if self[c.AGENT_PLACEHOLDER] and placeholder_obs is not None:
                obs_dict[c.AGENT_PLACEHOLDER] = placeholder_obs
            if self.parse_doors and door_obs is not None:
                obs_dict[c.DOORS] = door_obs[:]
            obs_dict.update(add_obs_dict)
            obsn = np.vstack(list(obs_dict.values()))
            if self.obs_prop.pomdp_r:
                obsn = self._do_pomdp_cutout(agent, obsn)

            raw_obs = self.per_agent_raw_observations_hook(agent)
            raw_obs = {key: np.expand_dims(val, 0) if val.ndim != 3 else val for key, val in raw_obs.items()}
            obsn = np.vstack((obsn, *raw_obs.values()))

            keys = list(chain(obs_dict.keys(), raw_obs.keys()))
            idxs = np.cumsum([x.shape[0] for x in chain(obs_dict.values(), raw_obs.values())]) - 1
            per_agent_expl_idx[agent.name] = {key: list(range(d, b)) for key, d, b in
                                              zip(keys, idxs, list(idxs[1:]) + [idxs[-1]+1, ])}

            # Shadow Casting
            if agent.step_result is not None:
                pass
            else:
                assert self._steps == 0
                agent.step_result = {'action_name': a.NOOP, 'action_valid': True,
                                     'collisions': [], 'lightmap': None}
            if self.obs_prop.cast_shadows:
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

                agent.step_result['lightmap'] = light_block_map

                obsn[shadowed_obs] = ((obsn[shadowed_obs] * light_block_map) + 0.) - (1 - light_block_map)
            else:
                if self._pomdp_r:
                    agent.step_result['lightmap'] = np.ones(self._obs_shape)
                else:
                    agent.step_result['lightmap'] = None

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
        ra, d = self._pomdp_r, self.pomdp_diameter
        x0, x1 = max(0, agent.x - ra), min(agent.x + ra + 1, self._level_shape[0])
        y0, y1 = max(0, agent.y - ra), min(agent.y + ra + 1, self._level_shape[1])
        oobs = obs_to_be_padded[:, x0:x1, y0:y1]
        if oobs.shape[1:] != (d, d):
            if xd := oobs.shape[1] % d:
                if agent.x > ra:
                    x0_pad = 0
                    x1_pad = (d - xd)
                else:
                    x0_pad = ra - agent.x
                    x1_pad = 0
            else:
                x0_pad, x1_pad = 0, 0

            if yd := oobs.shape[2] % d:
                if agent.y > ra:
                    y0_pad = 0
                    y1_pad = (d - yd)
                else:
                    y0_pad = ra - agent.y
                    y1_pad = 0
            else:
                y0_pad, y1_pad = 0, 0

            oobs = np.pad(oobs, ((0, 0), (x0_pad, x1_pad), (y0_pad, y1_pad)), 'constant')
        return oobs

    def get_all_tiles_with_collisions(self) -> List[Floor]:
        tiles = [x for x in self[c.FLOOR] if len(x.guests_that_can_collide) > 1]
        if False:
            tiles_with_collisions = list()
            for tile in self[c.FLOOR]:
                if tile.is_occupied():
                    guests = tile.guests_that_can_collide
                    if len(guests) >= 2:
                        tiles_with_collisions.append(tile)
        return tiles

    def _do_move_action(self, agent: Agent, action: Action) -> (dict, dict):
        info_dict = dict()
        new_tile, valid = self._check_agent_move(agent, action)
        if valid:
            # Does not collide width level boundaries
            valid = agent.move(new_tile)
            if valid:
                # This will spam your logs, beware!
                self.print(f'{agent.name} just moved {action.identifier} from {agent.last_pos} to {agent.pos}.')
                info_dict.update({f'{agent.name}_move': 1, 'move': 1})
                pass
            else:
                valid = c.NOT_VALID
                self.print(f'{agent.name} just hit the wall at {agent.pos}. ({action.identifier})')
                info_dict.update({f'{agent.name}_wall_collide': 1, 'wall_collide': 1})
        else:
            # Agent seems to be trying to Leave the level
            self.print(f'{agent.name} tried to leave the level {agent.pos}. ({action.identifier})')
            info_dict.update({f'{agent.name}_wall_collide': 1, 'wall_collide': 1})
        reward_value = self.rewards_base.MOVEMENTS_VALID if valid else self.rewards_base.MOVEMENTS_FAIL
        reward = {'value': reward_value, 'reason': action.identifier, 'info': info_dict}
        return valid, reward

    def _check_agent_move(self, agent, action: Action) -> (Floor, bool):
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
                        if door.is_closed:
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

    def build_reward_result(self, global_env_rewards: list) -> (int, dict):
        # Returns: Reward, Info
        info = defaultdict(lambda: 0.0)

        # Gather additional sub-env rewards and calculate collisions
        for agent in self[c.AGENT]:

            rewards = self.per_agent_reward_hook(agent)
            for reward in rewards:
                agent.step_result['rewards'].append(reward)
            if collisions := agent.step_result['collisions']:
                self.print(f't = {self._steps}\t{agent.name} has collisions with {collisions}')
                info[c.COLLISION] += 1
                reward = {'value': self.rewards_base.COLLISION,
                          'reason': c.COLLISION,
                          'info': {f'{agent.name}_{c.COLLISION}': 1}}
                agent.step_result['rewards'].append(reward)
            else:
                # No Collisions, nothing to do
                pass

        comb_rewards = {agent.name: sum(x['value'] for x in agent.step_result['rewards']) for agent in self[c.AGENT]}

        # Combine the per_agent_info_dict:
        combined_info_dict = defaultdict(lambda: 0)
        for agent in self[c.AGENT]:
            for reward in agent.step_result['rewards']:
                combined_info_dict.update(reward['info'])

        combined_info_dict = dict(combined_info_dict)
        combined_info_dict.update(info)

        global_reward_sum = sum(global_env_rewards)
        if self.individual_rewards:
            self.print(f"rewards are {comb_rewards}")
            reward = list(comb_rewards.values())
            reward = [x + global_reward_sum for x in reward]
            return reward, combined_info_dict
        else:
            reward = sum(comb_rewards.values()) + global_reward_sum
            self.print(f"reward is {reward}")
        return reward, combined_info_dict

    def start_recording(self):
        self._record_episodes = True

    def stop_recording(self):
        self._record_episodes = False

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
            agents.append(RenderEntity(name, agent.pos, 1, 'none', state, i + 1, agent.step_result['lightmap']))
        doors = []
        if self.parse_doors:
            for i, door in enumerate(self[c.DOORS]):
                name, state = 'door_open' if door.is_open else 'door_closed', 'blank'
                doors.append(RenderEntity(name, door.pos, 1, 'none', state, i + 1))
        additional_assets = self.render_assets_hook()

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
    @abc.abstractmethod
    def actions_hook(self) -> Union[Action, List[Action]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A list of Actions-object holding all additional actions.
        :rtype:             List[Action]
        """
        return []

    @property
    @abc.abstractmethod
    def entities_hook(self) -> Dict[(str, Entities)]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A single Entites collection or a list of such.
        :rtype:             Union[Entities, List[Entities]]
        """
        return {}

    # Functions which provide additions to functions of the base class
    #  Always call super!!!!!!
    @abc.abstractmethod
    def reset_hook(self) -> None:
        pass

    @abc.abstractmethod
    def pre_step_hook(self) -> None:
        pass

    @abc.abstractmethod
    def do_additional_actions(self, agent: Agent, action: Action) -> (bool, dict):
        return None

    @abc.abstractmethod
    def step_hook(self) -> (List[dict], dict):
        return [], {}

    @abc.abstractmethod
    def check_additional_done(self) -> (bool, dict):
        return False, {}

    @abc.abstractmethod
    def observations_hook(self) -> Dict[str, np.typing.ArrayLike]:
        return {}

    @abc.abstractmethod
    def per_agent_reward_hook(self, agent: Agent) -> Dict[str, dict]:
        return {}

    @abc.abstractmethod
    def post_step_hook(self) -> dict:
        return {}

    @abc.abstractmethod
    def per_agent_raw_observations_hook(self, agent) -> Dict[str, np.typing.ArrayLike]:
        additional_raw_observations = {}
        if self.obs_prop.show_global_position_info:
            global_pos_obs = np.zeros(self._obs_shape)
            global_pos_obs[:2, 0] = self[c.GLOBAL_POSITION].by_entity(agent).encoding
            additional_raw_observations.update({c.GLOBAL_POSITION: global_pos_obs})
        return additional_raw_observations

    @abc.abstractmethod
    def render_assets_hook(self):
        return []
