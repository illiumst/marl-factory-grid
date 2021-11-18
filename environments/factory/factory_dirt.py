import time
from enum import Enum
from typing import List, Union, NamedTuple, Dict
import random

import numpy as np

from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Entity, Tile
from environments.factory.base.registers import Entities, MovingEntityObjectRegister

from environments.factory.base.renderer import RenderEntity
from environments.utility_classes import ObservationProperties

CLEAN_UP_ACTION = h.EnvActions.CLEAN_UP


class DirtProperties(NamedTuple):
    initial_dirt_ratio: float = 0.3         # On INIT, on max how much tiles does the dirt spawn in percent.
    initial_dirt_spawn_r_var: float = 0.05   # How much does the dirt spawn amount vary?
    clean_amount: float = 1                 # How much does the robot clean with one actions.
    max_spawn_ratio: float = 0.20           # On max how much tiles does the dirt spawn in percent.
    max_spawn_amount: float = 0.3           # How much dirt does spawn per tile at max.
    spawn_frequency: int = 0                # Spawn Frequency in Steps.
    max_local_amount: int = 2               # Max dirt amount per tile.
    max_global_amount: int = 20             # Max dirt amount in the whole environment.
    dirt_smear_amount: float = 0.2          # Agents smear dirt, when not cleaning up in place.
    agent_can_interact: bool = True         # Whether the agents can interact with the dirt in this environment.
    done_when_clean = True


class Dirt(Entity):

    @property
    def can_collide(self):
        return False

    @property
    def amount(self):
        return self._amount

    def encoding(self):
        # Edit this if you want items to be drawn in the ops differntly
        return self._amount

    def __init__(self, *args, amount=None, **kwargs):
        super(Dirt, self).__init__(*args, **kwargs)
        self._amount = amount

    def set_new_amount(self, amount):
        self._amount = amount

    def summarize_state(self, **kwargs):
        state_dict = super().summarize_state(**kwargs)
        state_dict.update(amount=float(self.amount))
        return state_dict


class DirtRegister(MovingEntityObjectRegister):

    def as_array(self):
        if self._array is not None:
            self._array[:] = c.FREE_CELL.value
            for dirt in list(self.values()):
                if dirt.amount == 0:
                    self.delete_item(dirt)
                self._array[0, dirt.x, dirt.y] = dirt.amount
        else:
            self._array = np.zeros((1, *self._level_shape))
        return self._array

    _accepted_objects = Dirt

    @property
    def amount(self):
        return sum([dirt.amount for dirt in self])

    @property
    def dirt_properties(self):
        return self._dirt_properties

    def __init__(self, dirt_properties, *args):
        super(DirtRegister, self).__init__(*args)
        self._dirt_properties: DirtProperties = dirt_properties

    def spawn_dirt(self, then_dirty_tiles) -> c:
        if isinstance(then_dirty_tiles, Tile):
            then_dirty_tiles = [then_dirty_tiles]
        for tile in then_dirty_tiles:
            if not self.amount > self.dirt_properties.max_global_amount:
                dirt = self.by_pos(tile.pos)
                if dirt is None:
                    dirt = Dirt(tile, amount=self.dirt_properties.max_spawn_amount)
                    self.register_item(dirt)
                else:
                    new_value = dirt.amount + self.dirt_properties.max_spawn_amount
                    dirt.set_new_amount(min(new_value, self.dirt_properties.max_local_amount))
            else:
                return c.NOT_VALID
        return c.VALID

    def __repr__(self):
        s = super(DirtRegister, self).__repr__()
        return f'{s[:-1]}, {self.amount})'


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def entropy(x):
    return -(x * np.log(x + 1e-8)).sum()


# noinspection PyAttributeOutsideInit, PyAbstractClass
class DirtFactory(BaseFactory):

    @property
    def additional_actions(self) -> Union[Action, List[Action]]:
        super_actions = super().additional_actions
        if self.dirt_prop.agent_can_interact:
            super_actions.append(Action(enum_ident=CLEAN_UP_ACTION))
        return super_actions

    @property
    def additional_entities(self) -> Dict[(Enum, Entities)]:
        super_entities = super().additional_entities
        dirt_register = DirtRegister(self.dirt_prop, self._level_shape)
        super_entities.update(({c.DIRT: dirt_register}))
        return super_entities

    def __init__(self, *args, dirt_prop: DirtProperties = DirtProperties(), env_seed=time.time_ns(), **kwargs):
        if isinstance(dirt_prop, dict):
            dirt_prop = DirtProperties(**dirt_prop)
        self.dirt_prop = dirt_prop
        self._dirt_rng = np.random.default_rng(env_seed)
        self._dirt: DirtRegister
        kwargs.update(env_seed=env_seed)
        super().__init__(*args, **kwargs)

    def render_additional_assets(self, mode='human'):
        additional_assets = super().render_additional_assets()
        dirt = [RenderEntity('dirt', dirt.tile.pos, min(0.15 + dirt.amount, 1.5), 'scale')
                for dirt in self[c.DIRT]]
        additional_assets.extend(dirt)
        return additional_assets

    def clean_up(self, agent: Agent) -> c:
        if dirt := self[c.DIRT].by_pos(agent.pos):
            new_dirt_amount = dirt.amount - self.dirt_prop.clean_amount

            if new_dirt_amount <= 0:
                self[c.DIRT].delete_item(dirt)
            else:
                dirt.set_new_amount(max(new_dirt_amount, c.FREE_CELL.value))
            return c.VALID
        else:
            return c.NOT_VALID

    def trigger_dirt_spawn(self, initial_spawn=False):
        dirt_rng = self._dirt_rng
        free_for_dirt = [x for x in self[c.FLOOR]
                         if len(x.guests) == 0 or (len(x.guests) == 1 and isinstance(next(y for y in x.guests), Dirt))
                         ]
        self._dirt_rng.shuffle(free_for_dirt)
        if initial_spawn:
            var = self.dirt_prop.initial_dirt_spawn_r_var
            new_spawn = self.dirt_prop.initial_dirt_ratio + dirt_rng.uniform(-var, var)
        else:
            new_spawn = dirt_rng.uniform(0, self.dirt_prop.max_spawn_ratio)
        n_dirt_tiles = max(0, int(new_spawn * len(free_for_dirt)))
        self[c.DIRT].spawn_dirt(free_for_dirt[:n_dirt_tiles])

    def do_additional_step(self) -> dict:
        info_dict = super().do_additional_step()
        if smear_amount := self.dirt_prop.dirt_smear_amount:
            for agent in self[c.AGENT]:
                if agent.temp_valid and agent.last_pos != c.NO_POS:
                    if self._actions.is_moving_action(agent.temp_action):
                        if old_pos_dirt := self[c.DIRT].by_pos(agent.last_pos):
                            if smeared_dirt := round(old_pos_dirt.amount * smear_amount, 2):
                                old_pos_dirt.set_new_amount(max(0, old_pos_dirt.amount-smeared_dirt))
                                if new_pos_dirt := self[c.DIRT].by_pos(agent.pos):
                                    new_pos_dirt.set_new_amount(max(0, new_pos_dirt.amount + smeared_dirt))
                                else:
                                    if self[c.DIRT].spawn_dirt(agent.tile):
                                        new_pos_dirt = self[c.DIRT].by_pos(agent.pos)
                                        new_pos_dirt.set_new_amount(max(0, new_pos_dirt.amount + smeared_dirt))
        if self._next_dirt_spawn < 0:
            pass  # No Dirt Spawn
        elif not self._next_dirt_spawn:
            self.trigger_dirt_spawn()
            self._next_dirt_spawn = self.dirt_prop.spawn_frequency
        else:
            self._next_dirt_spawn -= 1
        return info_dict

    def do_additional_actions(self, agent: Agent, action: Action) -> Union[None, c]:
        valid = super().do_additional_actions(agent, action)
        if valid is None:
            if action == CLEAN_UP_ACTION:
                if self.dirt_prop.agent_can_interact:
                    valid = self.clean_up(agent)
                    return valid
                else:
                    return c.NOT_VALID
            else:
                return None
        else:
            return valid

    def do_additional_reset(self) -> None:
        super().do_additional_reset()
        self.trigger_dirt_spawn(initial_spawn=True)
        self._next_dirt_spawn = self.dirt_prop.spawn_frequency if self.dirt_prop.spawn_frequency else -1

    def check_additional_done(self):
        super_done = super().check_additional_done()
        done = self.dirt_prop.done_when_clean and (len(self[c.DIRT]) == 0)
        return super_done or done

    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        reward, info_dict = super().calculate_additional_reward(agent)
        dirt = [dirt.amount for dirt in self[c.DIRT]]
        current_dirt_amount = sum(dirt)
        dirty_tile_count = len(dirt)
        if dirty_tile_count:
            dirt_distribution_score = entropy(softmax(np.asarray(dirt)) / dirty_tile_count)
        else:
            dirt_distribution_score = 0

        info_dict.update(dirt_amount=current_dirt_amount)
        info_dict.update(dirty_tile_count=dirty_tile_count)
        info_dict.update(dirt_distribution_score=dirt_distribution_score)

        if agent.temp_action == CLEAN_UP_ACTION:
            if agent.temp_valid:
                # Reward if pickup succeds,
                #  0.5 on every pickup
                reward += 0.5
                if self.dirt_prop.done_when_clean and (len(self[c.DIRT]) == 0):
                    #  0.5 additional reward for the very last pickup
                    reward += 0.5
                self.print(f'{agent.name} did just clean up some dirt at {agent.pos}.')
                info_dict.update(dirt_cleaned=1)
            else:
                reward -= 0.01
                self.print(f'{agent.name} just tried to clean up some dirt at {agent.pos}, but failed.')
                info_dict.update({f'{agent.name}_failed_dirt_cleanup': 1})
                info_dict.update(failed_dirt_clean=1)

        # Potential based rewards ->
        #  track the last reward , minus the current reward = potential
        return reward, info_dict


if __name__ == '__main__':
    from environments.utility_classes import AgentRenderOptions as ARO
    render = True

    dirt_props = DirtProperties(1, 0.05, 0.1, 3, 1, 20, 0)

    obs_props = ObservationProperties(render_agents=ARO.COMBINED, omit_agent_self=True,
                                      pomdp_r=15, additional_agent_placeholder=None)

    move_props = {'allow_square_movement': True,
                  'allow_diagonal_movement': False,
                  'allow_no_op': False}

    factory = DirtFactory(n_agents=5, done_at_collision=False,
                          level_name='rooms', max_steps=400,
                          obs_prop=obs_props, parse_doors=True,
                          record_episodes=True, verbose=True,
                          mv_prop=move_props, dirt_prop=dirt_props
                          )

    # noinspection DuplicatedCode
    n_actions = factory.action_space.n - 1
    _ = factory.observation_space

    for epoch in range(4):
        random_actions = [[random.randint(0, n_actions) for _
                           in range(factory.n_agents)] for _
                          in range(factory.max_steps+1)]
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
pass
