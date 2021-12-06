import time
from collections import deque, UserList
from enum import Enum
from typing import List, Union, NamedTuple, Dict
import numpy as np
import random

from environments.factory.base.base_factory import BaseFactory
from environments.helpers import Constants as c
from environments import helpers as h
from environments.factory.base.objects import Agent, Entity, Action, Tile, MoveableEntity
from environments.factory.base.registers import Entities, EntityObjectRegister, ObjectRegister, \
    MovingEntityObjectRegister

from environments.factory.base.renderer import RenderEntity


NO_ITEM = 0
ITEM_DROP_OFF = 1


class Item(MoveableEntity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_despawn = -1

    @property
    def auto_despawn(self):
        return self._auto_despawn

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differently
        return 1

    def set_auto_despawn(self, auto_despawn):
        self._auto_despawn = auto_despawn


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

    def despawn_items(self, items: List[Item]):
        items = [items] if isinstance(items, Item) else items
        for item in items:
            del self[item]


class Inventory(UserList):

    @property
    def is_blocking_light(self):
        return False

    @property
    def name(self):
        return f'{self.__class__.__name__}({self.agent.name})'

    def __init__(self, pomdp_r: int, level_shape: (int, int), agent: Agent, capacity: int):
        super(Inventory, self).__init__()
        self.agent = agent
        self.pomdp_r = pomdp_r
        self._level_shape = level_shape
        if self.pomdp_r:
            self._array = np.zeros((1, pomdp_r * 2 + 1, pomdp_r * 2 + 1))
        else:
            self._array = np.zeros((1, *self._level_shape))
        self.capacity = min(capacity, self._array.size)

    def as_array(self):
        self._array[:] = c.FREE_CELL.value
        for item_idx, item in enumerate(self):
            x_diff, y_diff = divmod(item_idx, self._array.shape[1])
            self._array[0, int(x_diff), int(y_diff)] = item.encoding
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

    def summarize_state(self, **kwargs):
        attr_dict = {key: str(val) for key, val in self.__dict__.items() if not key.startswith('_') and key != 'data'}
        attr_dict.update(dict(items={val.name: val.summarize_state(**kwargs) for val in self}))
        attr_dict.update(dict(name=self.name))
        return attr_dict


class Inventories(ObjectRegister):

    _accepted_objects = Inventory
    is_blocking_light = False
    can_be_shadowed = False
    hide_from_obs_builder = True

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

    def summarize_states(self, n_steps=None):
        # as dict with additional nesting
        # return dict(items=super(Inventories, self).summarize_states())
        return super(Inventories, self).summarize_states(n_steps=n_steps)


class DropOffLocation(Entity):

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        return ITEM_DROP_OFF

    def __init__(self, *args, storage_size_until_full: int = 5, auto_item_despawn_interval: int = 5, **kwargs):
        super(DropOffLocation, self).__init__(*args, **kwargs)
        self.auto_item_despawn_interval = auto_item_despawn_interval
        self.storage = deque(maxlen=storage_size_until_full or None)

    def place_item(self, item: Item):
        if self.is_full:
            raise RuntimeWarning("There is currently no way to clear the storage or make it unfull.")
            return c.NOT_VALID
        else:
            self.storage.append(item)
            item.set_auto_despawn(self.auto_item_despawn_interval)
            return c.VALID

    @property
    def is_full(self):
        return False if not self.storage.maxlen else self.storage.maxlen == len(self.storage)

    def summarize_state(self, n_steps=None) -> dict:
        if n_steps == h.STEPS_START:
            return super().summarize_state(n_steps=n_steps)


class DropOffLocations(EntityObjectRegister):

    _accepted_objects = DropOffLocation

    def as_array(self):
        self._array[:] = c.FREE_CELL.value
        for item in self:
            if item.pos != c.NO_POS.value:
                self._array[0, item.x, item.y] = item.encoding
        return self._array

    def __repr__(self):
        super(DropOffLocations, self).__repr__()


class ItemProperties(NamedTuple):
    n_items:                   int  = 5     # How many items are there at the same time
    spawn_frequency:           int  = 10     # Spawn Frequency in Steps
    n_drop_off_locations:       int  = 5     # How many DropOff locations are there at the same time
    max_dropoff_storage_size:  int  = 0     # How many items are needed until the drop off is full
    max_agent_inventory_capacity:    int  = 5     # How many items are needed until the agent inventory is full
    agent_can_interact:        bool = True  # Whether agents have the possibility to interact with the domain items


# noinspection PyAttributeOutsideInit, PyAbstractClass
class ItemFactory(BaseFactory):
    # noinspection PyMissingConstructor
    def __init__(self, *args, item_prop: ItemProperties = ItemProperties(), env_seed=time.time_ns(), **kwargs):
        if isinstance(item_prop, dict):
            item_prop = ItemProperties(**item_prop)
        self.item_prop = item_prop
        kwargs.update(env_seed=env_seed)
        self._item_rng = np.random.default_rng(env_seed)
        assert (item_prop.n_items <= ((1 + kwargs.get('_pomdp_r', 0) * 2) ** 2)) or not kwargs.get('_pomdp_r', 0)
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

        empty_tiles = self[c.FLOOR].empty_tiles[:self.item_prop.n_drop_off_locations]
        drop_offs = DropOffLocations.from_tiles(
            empty_tiles, self._level_shape,
            entity_kwargs=dict(
                storage_size_until_full=self.item_prop.max_dropoff_storage_size)
        )
        item_register = ItemRegister(self._level_shape)
        empty_tiles = self[c.FLOOR].empty_tiles[:self.item_prop.n_items]
        item_register.spawn_items(empty_tiles)

        inventories = Inventories(self._level_shape if not self._pomdp_r else ((self.pomdp_diameter,) * 2))
        inventories.spawn_inventories(self[c.AGENT], self._pomdp_r,
                                      self.item_prop.max_agent_inventory_capacity)

        super_entities.update({c.DROP_OFF: drop_offs, c.ITEM: item_register, c.INVENTORY: inventories})
        return super_entities

    def additional_per_agent_obs_build(self, agent) -> List[np.ndarray]:
        additional_per_agent_obs_build = super().additional_per_agent_obs_build(agent)
        additional_per_agent_obs_build.append(self[c.INVENTORY].by_entity(agent).as_array())
        return additional_per_agent_obs_build

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
                if self.item_prop.agent_can_interact:
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
        self._next_item_spawn = self.item_prop.spawn_frequency
        self.trigger_item_spawn()

    def trigger_item_spawn(self):
        if item_to_spawns := max(0, (self.item_prop.n_items - len(self[c.ITEM]))):
            empty_tiles = self[c.FLOOR].empty_tiles[:item_to_spawns]
            self[c.ITEM].spawn_items(empty_tiles)
            self._next_item_spawn = self.item_prop.spawn_frequency
            self.print(f'{item_to_spawns} new items have been spawned; next spawn in {self._next_item_spawn}')
        else:
            self.print('No Items are spawning, limit is reached.')

    def do_additional_step(self) -> dict:
        # noinspection PyUnresolvedReferences
        info_dict = super().do_additional_step()
        for item in list(self[c.ITEM].values()):
            if item.auto_despawn >= 1:
                item.set_auto_despawn(item.auto_despawn-1)
            elif not item.auto_despawn:
                self[c.ITEM].delete_entity(item)
            else:
                pass

        if not self._next_item_spawn:
            self.trigger_item_spawn()
        else:
            self._next_item_spawn = max(0, self._next_item_spawn-1)
        return info_dict

    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        # noinspection PyUnresolvedReferences
        reward, info_dict = super().calculate_additional_reward(agent)
        if h.EnvActions.ITEM_ACTION == agent.temp_action:
            if agent.temp_valid:
                if drop_off := self[c.DROP_OFF].by_pos(agent.pos):
                    info_dict.update({f'{agent.name}_item_drop_off': 1})
                    info_dict.update(item_drop_off=1)
                    self.print(f'{agent.name} just dropped of an item at {drop_off.pos}.')
                    reward += 0.5
                else:
                    info_dict.update({f'{agent.name}_item_pickup': 1})
                    info_dict.update(item_pickup=1)
                    self.print(f'{agent.name} just picked up an item at {agent.pos}')
                    reward += 0.1
            else:
                if self[c.DROP_OFF].by_pos(agent.pos):
                    info_dict.update({f'{agent.name}_failed_drop_off': 1})
                    info_dict.update(failed_drop_off=1)
                    self.print(f'{agent.name} just tried to drop off at {agent.pos}, but failed.')
                    reward -= 0.1
                else:
                    info_dict.update({f'{agent.name}_failed_item_action': 1})
                    info_dict.update(failed_pick_up=1)
                    self.print(f'{agent.name} just tried to pick up an item at {agent.pos}, but failed.')
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
    from environments.utility_classes import AgentRenderOptions as ARO, ObservationProperties

    render = True

    item_probs = ItemProperties()

    obs_props = ObservationProperties(render_agents=ARO.LEVEL, omit_agent_self=True, pomdp_r=2)

    move_props = {'allow_square_movement': True,
                  'allow_diagonal_movement': False,
                  'allow_no_op': False}

    factory = ItemFactory(n_agents=3, done_at_collision=False,
                          level_name='rooms', max_steps=400,
                          obs_prop=obs_props, parse_doors=True,
                          record_episodes=True, verbose=True,
                          mv_prop=move_props, item_prop=item_probs
                          )

    # noinspection DuplicatedCode
    n_actions = factory.action_space.n - 1
    _ = factory.observation_space

    for epoch in range(4):
        random_actions = [[random.randint(0, n_actions) for _
                           in range(factory.n_agents)] for _
                          in range(factory.max_steps + 1)]
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
