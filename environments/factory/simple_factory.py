import time
from enum import Enum
from typing import List, Union, NamedTuple
import random

import numpy as np

from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Slice
from environments.factory.base.registers import Entities

from environments.factory.renderer import RenderEntity
from environments.utility_classes import MovementProperties


CLEAN_UP_ACTION = h.EnvActions.CLEAN_UP


class ObsSlice(Enum):
    OWN = -1
    LEVEL = c.LEVEL.value
    AGENT = c.AGENT.value


class DirtProperties(NamedTuple):
    clean_amount: int = 1               # How much does the robot clean with one actions.
    max_spawn_ratio: float = 0.2        # On max how much tiles does the dirt spawn in percent.
    gain_amount: float = 0.3            # How much dirt does spawn per tile.
    spawn_frequency: int = 5            # Spawn Frequency in Steps.
    max_local_amount: int = 2           # Max dirt amount per tile.
    max_global_amount: int = 20         # Max dirt amount in the whole environment.
    dirt_smear_amount: float = 0.2      # Agents smear dirt, when not cleaning up in place.
    agent_can_interact: bool = True     # Whether the agents can interact with the dirt in this environment.
    on_obs_slice: Enum = ObsSlice.LEVEL


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def entropy(x):
    return -(x * np.log(x + 1e-8)).sum()


# noinspection PyAttributeOutsideInit
class SimpleFactory(BaseFactory):

    @property
    def additional_actions(self) -> Union[Action, List[Action]]:
        super_actions = super(SimpleFactory, self).additional_actions
        if self.dirt_properties.agent_can_interact:
            super_actions.append(Action(CLEAN_UP_ACTION))
        return super_actions

    @property
    def additional_entities(self) -> Union[Entities, List[Entities]]:
        super_entities = super(SimpleFactory, self).additional_entities
        return super_entities

    @property
    def additional_slices(self) -> List[Slice]:
        super_slices = super(SimpleFactory, self).additional_slices
        super_slices.extend([Slice(c.DIRT, np.zeros(self._level_shape))])
        return super_slices

    def _is_clean_up_action(self, action: Union[str, Action, int]):
        if isinstance(action, int):
            action = self._actions[action]
        if isinstance(action, Action):
            action = action.name
        return action == CLEAN_UP_ACTION.name

    def __init__(self, *args, dirt_properties: DirtProperties = DirtProperties(), env_seed=time.time_ns(), **kwargs):
        self.dirt_properties = dirt_properties
        self._dirt_rng = np.random.default_rng(env_seed)
        kwargs.update(env_seed=env_seed)
        super(SimpleFactory, self).__init__(*args, **kwargs)

    def _flush_state(self):
        super(SimpleFactory, self)._flush_state()
        self._obs_cube[self._slices.get_idx(c.DIRT)] = self._slices.by_enum(c.DIRT).slice

    def render_additional_assets(self, mode='human'):
        additional_assets = super(SimpleFactory, self).render_additional_assets()
        dirt_slice = self._slices.by_enum(c.DIRT).slice
        dirt = [RenderEntity('dirt', tile.pos, min(0.15 + dirt_slice[tile.pos], 1.5), 'scale')
                for tile in [tile for tile in self._tiles if dirt_slice[tile.pos]]]
        additional_assets.extend(dirt)
        return additional_assets

    def spawn_dirt(self) -> None:
        dirt_slice = self._slices.by_enum(c.DIRT).slice
        # dirty_tiles = [tile for tile in self._tiles if dirt_slice[tile.pos]]
        curr_dirt_amount = dirt_slice.sum()
        if not curr_dirt_amount > self.dirt_properties.max_global_amount:
            free_for_dirt = self._tiles.empty_tiles

            # randomly distribute dirt across the grid
            new_spawn = self._dirt_rng.uniform(0, self.dirt_properties.max_spawn_ratio)
            n_dirt_tiles = max(0, int(new_spawn * len(free_for_dirt)))
            for tile in free_for_dirt[:n_dirt_tiles]:
                new_value = dirt_slice[tile.pos] + self.dirt_properties.gain_amount
                dirt_slice[tile.pos] = min(new_value, self.dirt_properties.max_local_amount)
        else:
            pass

    def clean_up(self, agent: Agent) -> bool:
        dirt_slice = self._slices.by_enum(c.DIRT).slice
        if old_dirt_amount := dirt_slice[agent.pos]:
            new_dirt_amount = old_dirt_amount - self.dirt_properties.clean_amount
            dirt_slice[agent.pos] = max(new_dirt_amount, c.FREE_CELL.value)
            return True
        else:
            return False

    def do_additional_step(self) -> dict:
        info_dict = super(SimpleFactory, self).do_additional_step()
        if smear_amount := self.dirt_properties.dirt_smear_amount:
            dirt_slice = self._slices.by_enum(c.DIRT).slice
            for agent in self._agents:
                if agent.temp_valid and agent.last_pos != c.NO_POS:
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
        return info_dict

    def do_additional_actions(self, agent: Agent, action: int) -> Union[None, bool]:
        valid = super(SimpleFactory, self).do_additional_actions(agent, action)
        if valid is None:
            if self._is_clean_up_action(action):
                if self.dirt_properties.agent_can_interact:
                    valid = self.clean_up(agent)
                    return valid
                else:
                    return False
            else:
                return None
        else:
            return valid

    def do_additional_reset(self) -> None:
        super(SimpleFactory, self).do_additional_reset()
        self.spawn_dirt()
        self._next_dirt_spawn = self.dirt_properties.spawn_frequency

    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        reward, info_dict = super(SimpleFactory, self).calculate_additional_reward(agent)
        dirt_slice = self._slices.by_enum(c.DIRT).slice
        dirty_tiles = [dirt_slice[tile.pos] for tile in self._tiles if dirt_slice[tile.pos]]
        current_dirt_amount = sum(dirty_tiles)
        dirty_tile_count = len(dirty_tiles)
        if dirty_tile_count:
            dirt_distribution_score = entropy(softmax(dirt_slice)) / dirty_tile_count
        else:
            dirt_distribution_score = 0

        info_dict.update(dirt_amount=current_dirt_amount)
        info_dict.update(dirty_tile_count=dirty_tile_count)
        info_dict.update(dirt_distribution_score=dirt_distribution_score)

        if agent.temp_collisions:
            self.print(f't = {self._steps}\t{agent.name} has collisions with {agent.temp_collisions}')

        if self._is_clean_up_action(agent.temp_action):
            if agent.temp_valid:
                reward += 0.5
                self.print(f'{agent.name} did just clean up some dirt at {agent.pos}.')
                info_dict.update(dirt_cleaned=1)
            else:
                reward -= 0.01
                self.print(f'{agent.name} just tried to clean up some dirt at {agent.pos}, but failed.')
                info_dict.update({f'{agent.name}_failed_action': 1})
                info_dict.update({f'{agent.name}_failed_action': 1})
                info_dict.update({f'{agent.name}_failed_dirt_cleanup': 1})

        # Potential based rewards ->
        #  track the last reward , minus the current reward = potential
        return reward, info_dict


if __name__ == '__main__':
    render = True

    dirt_props = DirtProperties(1, 0.05, 0.1, 3, 1, 20, 0.0)
    move_props = MovementProperties(True, True, False)

    factory = SimpleFactory(n_agents=1, done_at_collision=False, frames_to_stack=0,
                            level_name='rooms', max_steps=400,
                            omit_agent_slice_in_obs=True, parse_doors=True, pomdp_r=3,
                            record_episodes=False, verbose=False
                            )

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
