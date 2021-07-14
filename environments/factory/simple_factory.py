from typing import List, Union, NamedTuple
import random

import numpy as np

from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Object, Slice
from environments.factory.base.registers import Entities

from environments.factory.renderer import Renderer, Entity
from environments.utility_classes import MovementProperties

DIRT = "dirt"
CLEAN_UP_ACTION = 'clean_up'


class DirtProperties(NamedTuple):
    clean_amount: int = 1               # How much does the robot clean with one actions.
    max_spawn_ratio: float = 0.2        # On max how much tiles does the dirt spawn in percent.
    gain_amount: float = 0.3            # How much dirt does spawn per tile
    spawn_frequency: int = 5            # Spawn Frequency in Steps
    max_local_amount: int = 2           # Max dirt amount per tile.
    max_global_amount: int = 20         # Max dirt amount in the whole environment.
    dirt_smear_amount: float = 0.2      # Agents smear dirt, when not cleaning up in place


# noinspection PyAttributeOutsideInit
class SimpleFactory(BaseFactory):

    @property
    def additional_actions(self) -> List[Object]:
        return [Action(CLEAN_UP_ACTION)]

    @property
    def additional_entities(self) -> Union[Entities, List[Entities]]:
        return []

    @property
    def additional_slices(self) -> List[Slice]:
        return [Slice('dirt', np.zeros(self._level_shape))]

    def _is_clean_up_action(self, action: Union[str, int]):
        if isinstance(action, str):
            action = self._actions.by_name(action)
        return self._actions[action].name == CLEAN_UP_ACTION

    def __init__(self, *args, dirt_properties: DirtProperties = DirtProperties(), verbose=False, **kwargs):
        self.dirt_properties = dirt_properties
        self.verbose = verbose
        self._renderer = None  # expensive - don't use it when not required !
        super(SimpleFactory, self).__init__(*args, **kwargs)

    def _flush_state(self):
        super(SimpleFactory, self)._flush_state()
        self._obs_cube[self._slices.get_idx_by_name(DIRT)] = self._slices.by_name(DIRT).slice

    def render(self, mode='human'):

        if not self._renderer:  # lazy init
            height, width = self._obs_cube.shape[1:]
            self._renderer = Renderer(width, height, view_radius=self.pomdp_radius, fps=5)
        dirt_slice = self._slices.by_name(DIRT).slice
        dirt = [Entity('dirt', tile.pos, min(0.15 + dirt_slice[tile.pos], 1.5), 'scale')
                for tile in [tile for tile in self._tiles if dirt_slice[tile.pos]]]
        walls = [Entity('wall', pos)
                 for pos in np.argwhere(self._slices.by_enum(c.LEVEL).slice == c.OCCUPIED_CELL.value)]

        def asset_str(agent):
            # What does this abonimation do?
            # if any([x is None for x in [self._slices[j] for j in agent.collisions]]):
            #     print('error')
            col_names = [x.name for x in agent.temp_collisions]
            if c.AGENT.value in col_names:
                return 'agent_collision', 'blank'
            elif not agent.temp_valid or c.LEVEL.name in col_names or c.AGENT.name in col_names:
                return c.AGENT.value, 'invalid'
            elif self._is_clean_up_action(agent.temp_action):
                return c.AGENT.value, 'valid'
            else:
                return c.AGENT.value, 'idle'
        agents = []
        for i, agent in enumerate(self._agents):
            name, state = asset_str(agent)
            agents.append(Entity(name, agent.pos, 1, 'none', state, i+1))
        doors = []
        if self.parse_doors:
            for i, door in enumerate(self._doors):
                name, state = 'door_open' if door.is_open else 'door_closed', 'blank'
                agents.append(Entity(name, door.pos, 1, 'none', state, i+1))
        self._renderer.render(dirt+walls+agents+doors)

    def spawn_dirt(self) -> None:
        dirt_slice = self._slices.by_name(DIRT).slice
        # dirty_tiles = [tile for tile in self._tiles if dirt_slice[tile.pos]]
        curr_dirt_amount = dirt_slice.sum()
        if not curr_dirt_amount > self.dirt_properties.max_global_amount:
            free_for_dirt = self._tiles.empty_tiles

            # randomly distribute dirt across the grid
            n_dirt_tiles = int(random.uniform(0, self.dirt_properties.max_spawn_ratio) * len(free_for_dirt))
            for tile in free_for_dirt[:n_dirt_tiles]:
                new_value = dirt_slice[tile.pos] + self.dirt_properties.gain_amount
                dirt_slice[tile.pos] = min(new_value, self.dirt_properties.max_local_amount)
        else:
            pass

    def clean_up(self, agent: Agent) -> bool:
        dirt_slice = self._slices.by_name(DIRT).slice
        if dirt_slice[agent.pos]:
            new_dirt_amount = dirt_slice[agent.pos] - self.dirt_properties.clean_amount
            dirt_slice[agent.pos] = max(new_dirt_amount, c.FREE_CELL.value)
            return True
        else:
            return False

    def post_step(self) -> dict:
        if smear_amount := self.dirt_properties.dirt_smear_amount:
            dirt_slice = self._slices.by_name(DIRT).slice
            for agent in self._agents:
                if agent.temp_valid and agent.last_pos != h.NO_POS:
                    if dirt := dirt_slice[agent.last_pos]:
                        if smeared_dirt := round(dirt * smear_amount, 2):
                            dirt_slice[agent.last_pos] = max(0, dirt_slice[agent.last_pos]-smeared_dirt)
                            dirt_slice[agent.pos] = min((self.dirt_properties.max_local_amount,
                                                         dirt_slice[agent.pos] + smeared_dirt)
                                                        )

        if not self._next_dirt_spawn:
            self.spawn_dirt()
            self._next_dirt_spawn = self.dirt_properties.spawn_frequency
        else:
            self._next_dirt_spawn -= 1
        return {}

    def do_additional_actions(self, agent: Agent, action: int) -> bool:
        if action != self._actions.is_moving_action(action):
            if self._is_clean_up_action(action):
                valid = self.clean_up(agent)
                return valid
            else:
                raise RuntimeError('This should not happen!!!')
        else:
            raise RuntimeError('This should not happen!!!')

    def reset(self) -> (np.ndarray, int, bool, dict):
        _ = super().reset()  # state, reward, done, info ... =
        self.spawn_dirt()
        self._next_dirt_spawn = self.dirt_properties.spawn_frequency
        obs = self._get_observations()
        return obs

    def calculate_reward(self) -> (int, dict):
        info_dict = dict()

        dirt_slice = self._slices.by_name(DIRT).slice
        dirty_tiles = [dirt_slice[tile.pos] for tile in self._tiles if dirt_slice[tile.pos]]
        current_dirt_amount = sum(dirty_tiles)
        dirty_tile_count = len(dirty_tiles)

        info_dict.update(dirt_amount=current_dirt_amount)
        info_dict.update(dirty_tile_count=dirty_tile_count)

        try:
            # penalty = current_dirt_amount
            reward = 0
        except (ZeroDivisionError, RuntimeWarning):
            reward = 0

        for agent in self._agents:
            if agent.temp_collisions:
                self.print(f't = {self._steps}\t{agent.name} has collisions with {agent.temp_collisions}')

            if self._is_clean_up_action(agent.temp_action):
                if agent.temp_valid:
                    reward += 1
                    self.print(f'{agent.name} did just clean up some dirt at {agent.pos}.')
                    info_dict.update(dirt_cleaned=1)
                else:
                    reward -= 0.01
                    self.print(f'{agent.name} just tried to clean up some dirt at {agent.pos}, but failed.')
                    info_dict.update({f'{agent.name}_failed_action': 1})
                    info_dict.update({f'{agent.name}_failed_action': 1})
                    info_dict.update({f'{agent.name}_failed_dirt_cleanup': 1})

            elif self._actions.is_moving_action(agent.temp_action):
                if agent.temp_valid:
                    # info_dict.update(movement=1)
                    reward -= 0.00
                else:
                    # self.print('collision')
                    reward -= 0.01
                    self.print(f'{agent.name} just hit the wall at {agent.pos}.')
                    info_dict.update({f'{agent.name}_vs_LEVEL': 1})

            elif self._actions.is_door_usage(agent.temp_action):
                if agent.temp_valid:
                    self.print(f'{agent.name} did just use the door at {agent.pos}.')
                    info_dict.update(door_used=1)
                else:
                    reward -= 0.01
                    self.print(f'{agent.name} just tried to use a door at {agent.pos}, but failed.')
                    info_dict.update({f'{agent.name}_failed_action': 1})
                    info_dict.update({f'{agent.name}_failed_door_open': 1})

            else:
                info_dict.update(no_op=1)
                reward -= 0.00

            for other_agent in agent.temp_collisions:
                info_dict.update({f'{agent.name}_vs_{other_agent.name}': 1})

        self.print(f"reward is {reward}")
        # Potential based rewards ->
        #  track the last reward , minus the current reward = potential
        return reward, info_dict

    def print(self, string):
        if self.verbose:
            print(string)


if __name__ == '__main__':
    render = True

    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.3, max_global_amount=20,
                                max_local_amount=2, spawn_frequency=3, max_spawn_ratio=0.05,
                                dirt_smear_amount=0.0)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    factory = SimpleFactory(movement_properties=move_props, dirt_properties=dirt_props, n_agents=1,
                            combin_agent_slices_in_obs=False, level_name='rooms', parse_doors=True,
                            pomdp_radius=3)

    n_actions = factory.action_space.n - 1
    _ = factory.observation_space

    for epoch in range(100):
        random_actions = [[random.randint(0, n_actions) for _ in range(factory.n_agents)] for _ in range(200)]
        env_state = factory.reset()
        r = 0
        for agent_i_action in random_actions:
            env_state, step_r, done_bool, info_obj = factory.step(agent_i_action)
            r += step_r
            if render:
                factory.render()
            if done_bool:
                break
        print(f'Factory run {epoch} done, reward is:\n    {r}')
