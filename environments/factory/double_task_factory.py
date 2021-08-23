import time
from collections import deque
from enum import Enum
from typing import List, Union, NamedTuple
import numpy as np

from environments.factory.simple_factory import SimpleFactory
from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.objects import Agent, Slice, Entity, Action
from environments.factory.base.registers import Entities

from environments.factory.renderer import RenderEntity


PICK_UP = 'pick_up'
DROP_OFF = 'drop_off'
NO_ITEM = 0
ITEM_DROP_OFF = -1


def inventory_slice_name(agent_i):
    if isinstance(agent_i, int):
        return f'{c.INVENTORY.name}_{c.AGENT.value}#{agent_i}'
    else:
        return f'{c.INVENTORY.name}_{agent_i}'


class DropOffLocation(Entity):

    def __init__(self, *args, storage_size_until_full: int = 5, **kwargs):
        super(DropOffLocation, self).__init__(DROP_OFF, *args, **kwargs)
        self.storage = deque(maxlen=storage_size_until_full)

    def place_item(self, item):
        self.storage.append(item)
        return True

    @property
    def is_full(self):
        return self.storage.maxlen == len(self.storage)


class ItemProperties(NamedTuple):
    n_items:                   int  = 1     # How many items are there at the same time
    spawn_frequency:           int  = 5     # Spawn Frequency in Steps
    max_dropoff_storage_size:  int  = 5     # How many items are needed until the drop off is full
    max_agent_storage_size:    int  = 5     # How many items are needed until the agent inventory is full
    agent_can_interact:        bool = True  # Whether agents have the possibility to interact with the domain items


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
class DoubleTaskFactory(SimpleFactory):
    # noinspection PyMissingConstructor
    def __init__(self, item_properties: ItemProperties, *args, with_dirt=False, env_seed=time.time_ns(), **kwargs):
        self.item_properties = item_properties
        kwargs.update(env_seed=env_seed)
        self._item_rng = np.random.default_rng(env_seed)
        assert item_properties.n_items < kwargs.get('pomdp_r', 0) ** 2 or not kwargs.get('pomdp_r', 0)
        self._super = self.__class__ if with_dirt else SimpleFactory
        super(self._super, self).__init__(*args, **kwargs)

    @property
    def additional_actions(self) -> Union[Action, List[Action]]:
        super_actions = super(self._super, self).additional_actions
        super_actions.append(Action(h.EnvActions.ITEM_ACTION))
        return super_actions

    @property
    def additional_entities(self) -> Union[Entities, List[Entities]]:
        super_entities = super(self._super, self).additional_entities
        return super_entities

    @property
    def additional_slices(self) -> Union[Slice, List[Slice]]:
        super_slices = super(self._super, self).additional_slices
        super_slices.append(Slice(c.ITEM, np.zeros(self._level_shape)))
        super_slices.extend([Slice(inventory_slice_name(agent_i), np.zeros(self._level_shape), can_be_shadowed=False)
                             for agent_i in range(self.n_agents)])
        return super_slices

    def _flush_state(self):
        super(self._super, self)._flush_state()

        # Flush environmental item state
        slice_idx = self._slices.get_idx(c.ITEM)
        self._obs_cube[slice_idx] = self._slices[slice_idx].slice

        # Flush per agent inventory state
        for agent in self._agents:
            agent_slice_idx = self._slices.get_idx_by_name(inventory_slice_name(agent.name))
            self._slices[agent_slice_idx].slice[:] = 0
            if len(agent.inventory) > 0:
                max_x = self.pomdp_r if self.pomdp_r else self._level_shape[0]
                x, y = (0, 0) if not self.pomdp_r else (max(agent.x - max_x, 0), max(agent.y - max_x, 0))
                for item in agent.inventory:
                    x_diff, y_diff = divmod(item, max_x)
                    self._slices[agent_slice_idx].slice[int(x+x_diff), int(y+y_diff)] = item
            self._obs_cube[agent_slice_idx] = self._slices[agent_slice_idx].slice

    def _is_item_action(self, action):
        if isinstance(action, int):
            action = self._actions[action]
        if isinstance(action, Action):
            action = action.name
        return action == h.EnvActions.ITEM_ACTION.name

    def do_item_action(self, agent: Agent):
        item_slice = self._slices.by_enum(c.ITEM).slice

        if item := item_slice[agent.pos]:
            if item == ITEM_DROP_OFF:
                if agent.inventory:
                    valid = self._item_drop_off.place_item(agent.inventory.pop(0))
                    return valid
                else:
                    return c.NOT_VALID

            elif item != NO_ITEM:
                if len(agent.inventory) < self.item_properties.max_agent_storage_size:
                    agent.inventory.append(item_slice[agent.pos])
                    item_slice[agent.pos] = NO_ITEM
                else:
                    return c.NOT_VALID
            return c.VALID
        else:
            return c.NOT_VALID

    def do_additional_actions(self, agent: Agent, action: int) -> Union[None, bool]:
        valid = super(self._super, self).do_additional_actions(agent, action)
        if valid is None:
            if self._is_item_action(action):
                if self.item_properties.agent_can_interact:
                    valid = self.do_item_action(agent)
                    return bool(valid)
                else:
                    return False
            else:
                return None
        else:
            return valid

    def do_additional_reset(self) -> None:
        super(self._super, self).do_additional_reset()
        self.spawn_drop_off_location()
        self.spawn_items(self.item_properties.n_items)
        self._next_item_spawn = self.item_properties.spawn_frequency
        for agent in self._agents:
            agent.inventory = list()

    def do_additional_step(self) -> dict:
        info_dict = super(self._super, self).do_additional_step()
        if not self._next_item_spawn:
            if item_to_spawn := (self.item_properties.n_items -
                                 (np.sum(self._slices.by_enum(c.ITEM).slice.astype(bool)) - 1)):
                self.spawn_items(item_to_spawn)
                self._next_item_spawn = self.item_properties.spawn_frequency
            else:
                self.print('No Items are spawning, limit is reached.')
        else:
            self._next_item_spawn -= 1
        return info_dict

    def spawn_drop_off_location(self):
        single_empty_tile = self._tiles.empty_tiles[0]
        self._item_drop_off = DropOffLocation(single_empty_tile,
                                              storage_size_until_full=self.item_properties.max_dropoff_storage_size)
        single_empty_tile.enter(self._item_drop_off)
        self._slices.by_enum(c.ITEM).slice[single_empty_tile.pos] = ITEM_DROP_OFF

    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        reward, info_dict = super(self._super, self).calculate_additional_reward(agent)
        if self._is_item_action(agent.temp_action):
            if agent.temp_valid:
                if agent.pos == self._item_drop_off.pos:
                    info_dict.update({f'{agent.name}_item_dropoff': 1})

                    reward += 1
                else:
                    info_dict.update({f'{agent.name}_item_pickup': 1})
                    reward += 0.1
            else:
                info_dict.update({f'{agent.name}_failed_item_action': 1})
                reward -= 0.1
        return reward, info_dict

    def render_additional_assets(self, mode='human'):
        additional_assets = super(self._super, self).render_additional_assets()
        item_slice = self._slices.by_enum(c.ITEM).slice
        items = [RenderEntity(DROP_OFF if item_slice[tile.pos] == ITEM_DROP_OFF else c.ITEM.value, tile.pos)
                 for tile in [tile for tile in self._tiles if item_slice[tile.pos] != NO_ITEM]]
        additional_assets.extend(items)
        return additional_assets

    def spawn_items(self, n_items):
        tiles = self._tiles.empty_tiles[:n_items]
        item_slice = self._slices.by_enum(c.ITEM).slice
        for idx, tile in enumerate(tiles, start=1):
            item_slice[tile.pos] = idx
        pass


if __name__ == '__main__':
    import random
    render = True

    item_props = ItemProperties()

    factory = DoubleTaskFactory(item_props, n_agents=1, done_at_collision=False, frames_to_stack=0,
                                level_name='rooms', max_steps=400,
                                omit_agent_slice_in_obs=True, parse_doors=True, pomdp_r=3,
                                record_episodes=False, verbose=False
                                )

    n_actions = factory.action_space.n - 1
    _ = factory.observation_space

    for epoch in range(100):
        random_actions = [[random.randint(0, n_actions) for _ in range(factory.n_agents)] for _ in range(200)]
        env_state = factory.reset()
        rew = 0
        for agent_i_action in random_actions:
            env_state, step_r, done_bool, info_obj = factory.step(agent_i_action)
            rew += step_r
            if render:
                factory.render()
            if done_bool:
                break
        print(f'Factory run {epoch} done, reward is:\n    {rew}')
