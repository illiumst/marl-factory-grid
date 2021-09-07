import time
from collections import deque, UserList
from enum import Enum
from typing import List, Union, NamedTuple, Dict
import numpy as np

from environments.factory.base.base_factory import BaseFactory
from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.objects import Agent, Entity, Action, Tile, MoveableEntity
from environments.factory.base.registers import Entities, EntityObjectRegister, ObjectRegister, \
    MovingEntityObjectRegister

from environments.factory.renderer import RenderEntity


NO_ITEM = 0
ITEM_DROP_OFF = 1


def inventory_slice_name(agent_i):
    if isinstance(agent_i, int):
        return f'{c.INVENTORY.name}_{c.AGENT.value}#{agent_i}'
    else:
        return f'{c.INVENTORY.name}_{agent_i}'


class Item(MoveableEntity):

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differntly
        return 1


class ItemRegister(MovingEntityObjectRegister):

    def as_array(self):
        self._array[:] = c.FREE_CELL.value
        for item in self:
            if item.pos != c.NO_POS.value:
                self._array[0, item.x, item.y] = item.encoding
        return self._array

    _accepted_objects = Item

    def spawn_items(self, tiles: List[Tile]):
        items = [Item(tile) for tile in tiles]
        self.register_additional_items(items)


class Inventory(UserList):

    @property
    def is_blocking_light(self):
        return False

    @property
    def name(self):
        return self.agent.name

    def __init__(self, pomdp_r: int, level_shape: (int, int), agent: Agent, capacity: int):
        super(Inventory, self).__init__()
        self.agent = agent
        self.capacity = capacity
        self.pomdp_r = pomdp_r
        self._level_shape = level_shape
        self._array = np.zeros((1, *self._level_shape))

    def as_array(self):
        self._array[:] = c.FREE_CELL.value
        max_x = self.pomdp_r * 2 + 1 if self.pomdp_r else self._level_shape[0]
        if self.pomdp_r:
            x, y = max(self.agent.x - self.pomdp_r, 0), max(self.agent.y - self.pomdp_r, 0)
        else:
            x, y = (0, 0)

        for item_idx, item in enumerate(self):
            x_diff, y_diff = divmod(item_idx, max_x)
            self._array[0, int(x + x_diff), int(y + y_diff)] = item.encoding
        return self._array

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.agent.name}]({self.data})'

    def append(self, item) -> None:
        if len(self) < self.capacity:
            super(Inventory, self).append(item)
        else:
            raise RuntimeError('Inventory is full')

    def belongs_to_entity(self, entity):
        return self.agent == entity

    def summarize_state(self):
        return {val.name: val.summarize_state() for val in self}


class Inventories(ObjectRegister):

    _accepted_objects = Inventory
    is_blocking_light = False
    can_be_shadowed = False

    def __init__(self, *args, **kwargs):
        super(Inventories, self).__init__(*args, is_per_agent=True, individual_slices=True, **kwargs)
        self.is_observable = True

    def as_array(self):
        # self._array[:] = c.FREE_CELL.value
        for inv_idx, inventory in enumerate(self):
            self._array[inv_idx] = inventory.as_array()
        return self._array

    def spawn_inventories(self, agents, pomdp_r, capacity):
        inventories = [self._accepted_objects(pomdp_r, self._level_shape, agent, capacity)
                       for _, agent in enumerate(agents)]
        self.register_additional_items(inventories)

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, inv in enumerate(self) if inv.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def by_entity(self, entity):
        try:
            return next((inv for inv in self if inv.belongs_to_entity(entity)))
        except StopIteration:
            return None


class DropOffLocation(Entity):

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        return ITEM_DROP_OFF

    def __init__(self, *args, storage_size_until_full: int = 5, **kwargs):
        super(DropOffLocation, self).__init__(*args, **kwargs)
        self.storage = deque(maxlen=storage_size_until_full or None)

    def place_item(self, item):
        if self.is_full:
            raise RuntimeWarning("There is currently no way to clear the storage or make it unfull.")
            return c.NOT_VALID
        else:
            self.storage.append(item)
            return c.VALID

    @property
    def is_full(self):
        return False if not self.storage.maxlen else self.storage.maxlen == len(self.storage)


class DropOffLocations(EntityObjectRegister):

    _accepted_objects = DropOffLocation

    def as_array(self):
        self._array[:] = c.FREE_CELL.value
        for item in self:
            if item.pos != c.NO_POS.value:
                self._array[0, item.x, item.y] = item.encoding
        return self._array


class ItemProperties(NamedTuple):
    n_items:                   int  = 5     # How many items are there at the same time
    spawn_frequency:           int  = 5     # Spawn Frequency in Steps
    n_drop_off_locations:       int  = 5     # How many DropOff locations are there at the same time
    max_dropoff_storage_size:  int  = 0     # How many items are needed until the drop off is full
    max_agent_inventory_capacity:    int  = 5     # How many items are needed until the agent inventory is full
    agent_can_interact:        bool = True  # Whether agents have the possibility to interact with the domain items


# noinspection PyAttributeOutsideInit, PyAbstractClass
class ItemFactory(BaseFactory):
    # noinspection PyMissingConstructor
    def __init__(self, *args, item_properties: ItemProperties = ItemProperties(),  env_seed=time.time_ns(), **kwargs):
        if isinstance(item_properties, dict):
            item_properties = ItemProperties(**item_properties)
        self.item_properties = item_properties
        kwargs.update(env_seed=env_seed)
        self._item_rng = np.random.default_rng(env_seed)
        assert (item_properties.n_items <= ((1 + kwargs.get('pomdp_r', 0) * 2) ** 2)) or not kwargs.get('pomdp_r', 0)
        super().__init__(*args, **kwargs)

    @property
    def additional_actions(self) -> Union[Action, List[Action]]:
        # noinspection PyUnresolvedReferences
        super_actions = super().additional_actions
        super_actions.append(Action(enum_ident=h.EnvActions.ITEM_ACTION))
        return super_actions

    @property
    def additional_entities(self) -> Dict[(Enum, Entities)]:
        # noinspection PyUnresolvedReferences
        super_entities = super().additional_entities

        empty_tiles = self[c.FLOOR].empty_tiles[:self.item_properties.n_drop_off_locations]
        drop_offs = DropOffLocations.from_tiles(empty_tiles, self._level_shape,
                                                storage_size_until_full=self.item_properties.max_dropoff_storage_size)
        item_register = ItemRegister(self._level_shape)
        empty_tiles = self[c.FLOOR].empty_tiles[:self.item_properties.n_items]
        item_register.spawn_items(empty_tiles)

        inventories = Inventories(self._level_shape)
        inventories.spawn_inventories(self[c.AGENT], self.pomdp_r,
                                      self.item_properties.max_agent_inventory_capacity)

        super_entities.update({c.DROP_OFF: drop_offs, c.ITEM: item_register, c.INVENTORY: inventories})
        return super_entities

    def do_item_action(self, agent: Agent):
        inventory = self[c.INVENTORY].by_entity(agent)
        if drop_off := self[c.DROP_OFF].by_pos(agent.pos):
            if inventory:
                valid = drop_off.place_item(inventory.pop(0))
                return valid
            else:
                return c.NOT_VALID
        elif item := self[c.ITEM].by_pos(agent.pos):
            try:
                inventory.append(item)
                item.move(self._NO_POS_TILE)
                return c.VALID
            except RuntimeError:
                return c.NOT_VALID
        else:
            return c.NOT_VALID

    def do_additional_actions(self, agent: Agent, action: Action) -> Union[None, c]:
        # noinspection PyUnresolvedReferences
        valid = super().do_additional_actions(agent, action)
        if valid is None:
            if action == h.EnvActions.ITEM_ACTION:
                if self.item_properties.agent_can_interact:
                    valid = self.do_item_action(agent)
                    return valid
                else:
                    return c.NOT_VALID
            else:
                return None
        else:
            return valid

    def do_additional_reset(self) -> None:
        # noinspection PyUnresolvedReferences
        super().do_additional_reset()
        self._next_item_spawn = self.item_properties.spawn_frequency
        self.trigger_item_spawn()

    def trigger_item_spawn(self):
        if item_to_spawns := max(0, (self.item_properties.n_items - len(self[c.ITEM]))):
            empty_tiles = self[c.FLOOR].empty_tiles[:item_to_spawns]
            self[c.ITEM].spawn_items(empty_tiles)
            self._next_item_spawn = self.item_properties.spawn_frequency
            self.print(f'{item_to_spawns} new items have been spawned; next spawn in {self._next_item_spawn}')
        else:
            self.print('No Items are spawning, limit is reached.')

    def do_additional_step(self) -> dict:
        # noinspection PyUnresolvedReferences
        info_dict = super().do_additional_step()
        if not self._next_item_spawn:
            self.trigger_item_spawn()
        else:
            self._next_item_spawn -= 1
        return info_dict

    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        # noinspection PyUnresolvedReferences
        reward, info_dict = super().calculate_additional_reward(agent)
        if h.EnvActions.ITEM_ACTION == agent.temp_action:
            if agent.temp_valid:
                if self[c.DROP_OFF].by_pos(agent.pos):
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
        # noinspection PyUnresolvedReferences
        additional_assets = super().render_additional_assets()
        items = [RenderEntity(c.ITEM.value, item.tile.pos) for item in self[c.ITEM]]
        additional_assets.extend(items)
        drop_offs = [RenderEntity(c.DROP_OFF.value, drop_off.tile.pos) for drop_off in self[c.DROP_OFF]]
        additional_assets.extend(drop_offs)
        return additional_assets


if __name__ == '__main__':
    import random
    render = True

    item_props = ItemProperties()

    factory = ItemFactory(item_properties=item_props, n_agents=3, done_at_collision=False, frames_to_stack=0,
                          level_name='rooms', max_steps=4000,
                          omit_agent_in_obs=True, parse_doors=True, pomdp_r=3,
                          record_episodes=False, verbose=False
                          )

    # noinspection DuplicatedCode
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
