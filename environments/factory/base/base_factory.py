from pathlib import Path
from typing import List, Union, Iterable

import gym
import numpy as np
from gym import spaces

import yaml
from gym.wrappers import FrameStack

from environments.helpers import Constants as c, Constants
from environments import helpers as h
from environments.factory.base.objects import Slice, Agent, Tile, Action
from environments.factory.base.registers import StateSlices, Actions, Entities, Agents, Doors, FloorTiles
from environments.utility_classes import MovementProperties

REC_TAC = 'rec'


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
            shape = (self._obs_cube.shape[0] - agent_slice, self.pomdp_radius * 2 + 1, self.pomdp_radius * 2 + 1)
            space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
            return space
        else:
            shape = [x-agent_slice if idx == 0 else x for idx, x in enumerate(self._obs_cube.shape)]
            space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
            return space

    @property
    def pomdp_diameter(self):
        return self.pomdp_radius * 2 + 1

    @property
    def movement_actions(self):
        return self._actions.movement_actions

    @property
    def additional_actions(self) -> Union[str, List[str]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A list of Actions-object holding all additional actions.
        :rtype:             List[Action]
        """
        raise NotImplementedError('Please register additional actions ')

    @property
    def additional_entities(self) -> Union[Entities, List[Entities]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A single Entites collection or a list of such.
        :rtype:             Union[Entities, List[Entities]]
        """
        raise NotImplementedError('Please register additional entities.')

    @property
    def additional_slices(self) -> Union[Slice, List[Slice]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A list of Slice-objects.
        :rtype:             List[Slice]
        """
        raise NotImplementedError('Please register additional slices.')

    def __enter__(self):
        return self if self.frames_to_stack == 0 else FrameStack(self, self.frames_to_stack)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, level_name='simple', n_agents=1, max_steps=int(5e2), pomdp_radius: Union[None, int] = 0,
                 movement_properties: MovementProperties = MovementProperties(), parse_doors=False,
                 combin_agent_slices_in_obs: bool = False, frames_to_stack=0, record_episodes=False,
                 omit_agent_slice_in_obs=False, done_at_collision=False, **kwargs):
        assert frames_to_stack != 1 and frames_to_stack >= 0, "'frames_to_stack' cannot be negative or 1."

        # Attribute Assignment
        self.movement_properties = movement_properties
        self.level_name = level_name
        self._level_shape = None

        self.n_agents = n_agents
        self.max_steps = max_steps
        self.pomdp_radius = pomdp_radius
        self.combin_agent_slices_in_obs = combin_agent_slices_in_obs
        self.omit_agent_slice_in_obs = omit_agent_slice_in_obs
        self.frames_to_stack = frames_to_stack

        self.done_at_collision = done_at_collision
        self.record_episodes = record_episodes
        self.parse_doors = parse_doors

        # Actions
        self._actions = Actions(self.movement_properties, can_use_doors=self.parse_doors)
        if additional_actions := self.additional_actions:
            self._actions.register_additional_items(additional_actions)

        self.reset()

    def _init_state_slices(self) -> StateSlices:
        state_slices = StateSlices()

        # Objects
        # Level
        level_filepath = Path(__file__).parent.parent / h.LEVELS_DIR / f'{self.level_name}.txt'
        parsed_level = h.parse_level(level_filepath)
        level = [Slice(c.LEVEL.name, h.one_hot_level(parsed_level))]
        self._level_shape = level[0].shape

        # Doors
        parsed_doors = h.one_hot_level(parsed_level, c.DOOR)
        doors = [Slice(c.DOORS.name, parsed_doors)] if parsed_doors.any() and self.parse_doors else []

        # Agents
        agents = []
        for i in range(self.n_agents):
            agents.append(Slice(f'{c.AGENT.name}#{i}', np.zeros_like(level[0].slice)))
        state_slices.register_additional_items(level+doors+agents)

        # Additional Slices from SubDomains
        if additional_slices := self.additional_slices:
            state_slices.register_additional_items(additional_slices)
        return state_slices

    def _init_obs_cube(self) -> np.ndarray:
        x, y = self._slices.by_enum(c.LEVEL).shape
        state = np.zeros((len(self._slices), x, y))
        state[0] = self._slices.by_enum(c.LEVEL).slice
        if r := self.pomdp_radius:
            self._padded_obs_cube = np.full((len(self._slices), x + r*2, y + r*2), c.FREE_CELL.value)
            self._padded_obs_cube[0] = c.OCCUPIED_CELL.value
            self._padded_obs_cube[:, r:r+x, r:r+y] = state
        return state

    def _init_entities(self):
        # Tile Init
        self._tiles = FloorTiles.from_argwhere_coordinates(self._slices.by_enum(c.LEVEL).free_tiles)

        # Door Init
        if self.parse_doors:
            tiles = [self._tiles.by_pos(x) for x in self._slices.by_enum(c.DOORS).occupied_tiles]
            self._doors = Doors.from_tiles(tiles, context=self._tiles)

        # Agent Init on random positions
        self._agents = Agents.from_tiles(np.random.choice(self._tiles, self.n_agents))
        entities = Entities()
        entities.register_additional_items([self._agents])

        if self.parse_doors:
            entities.register_additional_items([self._doors])

        if additional_entities := self.additional_entities:
            entities.register_additional_items([additional_entities])

        return entities

    def reset(self) -> (np.ndarray, int, bool, dict):
        self._slices = self._init_state_slices()
        self._obs_cube = self._init_obs_cube()
        self._entitites = self._init_entities()
        self._flush_state()
        self._steps = 0

        info = self._summarize_state() if self.record_episodes else {}
        return None, None, None, info

    def pre_step(self) -> None:
        pass

    def post_step(self) -> dict:
        pass

    def step(self, actions):
        actions = [actions] if isinstance(actions, int) or np.isscalar(actions) else actions
        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self._steps += 1
        done = False

        # Pre step Hook for later use
        self.pre_step()

        # Move this in a seperate function?
        for action, agent in zip(actions, self._agents):
            agent.clear_temp_sate()
            action_name = self._actions[action]
            if self._actions.is_moving_action(action):
                valid = self._move_or_colide(agent, action_name)
            elif self._actions.is_no_op(action):
                valid = c.VALID.value
            elif self._actions.is_door_usage(action):
                # Check if agent raly stands on a door:
                if door := self._doors.by_pos(agent.pos):
                    door.use()
                    valid = c.VALID.value
                # When he doesn't...
                else:
                    valid = c.NOT_VALID.value
            else:
                valid = self.do_additional_actions(agent, action)
            agent.temp_action = action
            agent.temp_valid = valid

        self._flush_state()

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
            self._doors.tick_doors()

        # Finalize
        reward, info = self.calculate_reward()
        if self._steps >= self.max_steps:
            done = True
        info.update(step_reward=reward, step=self._steps)
        if self.record_episodes:
            info.update(self._summarize_state())

        # Post step Hook for later use
        info.update(self.post_step())

        obs = self._get_observations()

        return obs, reward, done, info

    def _flush_state(self):
        self._obs_cube[np.arange(len(self._slices)) != self._slices.get_idx(c.LEVEL)] = c.FREE_CELL.value
        if self.parse_doors:
            for door in self._doors:
                if door.is_open:
                    self._obs_cube[self._slices.get_idx(c.DOORS)][door.pos] = c.IS_OPEN_DOOR.value
                else:
                    self._obs_cube[self._slices.get_idx(c.DOORS)][door.pos] = c.IS_CLOSED_DOOR.value
        for agent in self._agents:
            self._obs_cube[self._slices.get_idx_by_name(agent.name)][agent.pos] = c.OCCUPIED_CELL.value
            if agent.last_pos != h.NO_POS:
                self._obs_cube[self._slices.get_idx_by_name(agent.name)][agent.last_pos] = c.FREE_CELL.value

    def _get_observations(self) -> np.ndarray:
        if self.n_agents == 1:
            obs = self._build_per_agent_obs(self._agents[0])
        elif self.n_agents >= 2:
            obs = np.stack([self._build_per_agent_obs(agent) for agent in self._agents])
        else:
            raise ValueError('n_agents cannot be smaller than 1!!')
        return obs

    def _build_per_agent_obs(self, agent: Agent) -> np.ndarray:
        first_agent_slice = self._slices.AGENTSTARTIDX
        if r := self.pomdp_radius:
            x, y = self._level_shape
            self._padded_obs_cube[:, r:r + x, r:r + y] = self._obs_cube
            global_x, global_y = agent.pos
            global_x += r
            global_y += r
            x0, x1 = max(0, global_x - self.pomdp_radius), global_x + self.pomdp_radius + 1
            y0, y1 = max(0, global_y - self.pomdp_radius), global_y + self.pomdp_radius + 1
            obs = self._padded_obs_cube[:, x0:x1, y0:y1]
        else:
            obs = self._obs_cube

        if self.combin_agent_slices_in_obs and self.n_agents > 1:
            agent_obs = np.sum(obs[[key for key, slice in self._slices.items() if c.AGENT.name in slice.name and
                                    (not self.omit_agent_slice_in_obs and slice.name != agent.name)]],
                               axis=0, keepdims=True)
            obs = np.concatenate((obs[:first_agent_slice], agent_obs, obs[first_agent_slice+self.n_agents:]))
            return obs
        else:
            if self.omit_agent_slice_in_obs:
                obs_new = obs[[key for key, val in self._slices.items() if c.AGENT.value not in val.name]]
                return obs_new
            else:
                return obs

    def do_additional_actions(self, agent_i: int, action: int) -> bool:
        raise NotImplementedError

    def get_all_tiles_with_collisions(self) -> List[Tile]:
        tiles_with_collisions = list()
        for tile in self._tiles:
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
        x_diff, y_diff = h.ACTIONMAP[action.name]
        x_new = agent.x + x_diff
        y_new = agent.y + y_diff

        new_tile = self._tiles.by_pos((x_new, y_new))
        if new_tile:
            valid = c.VALID
        else:
            tile = agent.tile
            valid = c.VALID
            return tile, valid

        if self.parse_doors and agent.last_pos != h.NO_POS:
            if door := self._doors.by_pos(agent.pos):
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

    def _summarize_state(self):
        summary = {f'{REC_TAC}_step': self._steps}
        for entity in self._entitites:
            if hasattr(entity, 'summarize_state'):
                summary.update({f'{REC_TAC}_{entity.name}': entity.summarize_state()})
        return summary
