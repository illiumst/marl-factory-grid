import time
from collections import deque
from typing import List, Union, NamedTuple, Dict
import numpy as np
import random

from environments.factory.base.base_factory import BaseFactory
from environments.helpers import Constants as BaseConstants
from environments.helpers import EnvActions as BaseActions
from environments import helpers as h
from environments.factory.base.objects import Agent, Entity, Action, Floor
from environments.factory.base.registers import Entities, EntityRegister, BoundEnvObjRegister, ObjectRegister

from environments.factory.base.renderer import RenderEntity


class Constants(BaseConstants):
    NO_ITEM = 0
    ITEM_DROP_OFF = 1
    # Item Env
    ITEM                = 'Item'
    INVENTORY           = 'Inventory'
    DROP_OFF            = 'Drop_Off'


class Actions(BaseActions):
    ITEM_ACTION     = 'ITEMACTION'


class RewardsItem(NamedTuple):
    DROP_OFF_VALID: float = 0.1
    DROP_OFF_FAIL: float = -0.1
    PICK_UP_FAIL: float  = -0.1
    PICK_UP_VALID: float  = 0.1


class Item(Entity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_despawn = -1

    @property
    def auto_despawn(self):
        return self._auto_despawn

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differently
        return 1

    def set_auto_despawn(self, auto_despawn):
        self._auto_despawn = auto_despawn

    def set_tile_to(self, no_pos_tile):
        assert self._register.__class__.__name__ != ItemRegister.__class__
        self._tile = no_pos_tile


class ItemRegister(EntityRegister):

    _accepted_objects = Item

    def spawn_items(self, tiles: List[Floor]):
        items = [Item(tile, self) for tile in tiles]
        self.register_additional_items(items)

    def despawn_items(self, items: List[Item]):
        items = [items] if isinstance(items, Item) else items
        for item in items:
            del self[item]


class Inventory(BoundEnvObjRegister):

    @property
    def name(self):
        return f'{self.__class__.__name__}({self._bound_entity.name})'

    def __init__(self, agent: Agent, capacity: int, *args, **kwargs):
        super(Inventory, self).__init__(agent, *args,  is_blocking_light=False, can_be_shadowed=False,  **kwargs)
        self.capacity = capacity

    def as_array(self):
        if self._array is None:
            self._array = np.zeros((1, *self._shape))
        return super(Inventory, self).as_array()

    def summarize_states(self, **kwargs):
        attr_dict = {key: str(val) for key, val in self.__dict__.items() if not key.startswith('_') and key != 'data'}
        attr_dict.update(dict(items={key: val.summarize_state(**kwargs) for key, val in self.items()}))
        attr_dict.update(dict(name=self.name))
        return attr_dict

    def pop(self):
        item_to_pop = self[0]
        self.delete_env_object(item_to_pop)
        return item_to_pop


class Inventories(ObjectRegister):

    _accepted_objects = Inventory
    is_blocking_light = False
    can_be_shadowed = False

    def __init__(self, obs_shape, *args, **kwargs):
        super(Inventories, self).__init__(*args, is_per_agent=True, individual_slices=True, **kwargs)
        self._obs_shape = obs_shape

    def as_array(self):
        return np.stack([inventory.as_array() for inv_idx, inventory in enumerate(self)])

    def spawn_inventories(self, agents, capacity):
        inventories = [self._accepted_objects(agent, capacity, self._obs_shape)
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

    def summarize_states(self, **kwargs):
        return {key: val.summarize_states(**kwargs) for key, val in self.items()}


class DropOffLocation(Entity):

    @property
    def encoding(self):
        return Constants.ITEM_DROP_OFF

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


class DropOffLocations(EntityRegister):

    _accepted_objects = DropOffLocation


class ItemProperties(NamedTuple):
    n_items:                         int  = 5     # How many items are there at the same time
    spawn_frequency:                 int  = 10     # Spawn Frequency in Steps
    n_drop_off_locations:            int  = 5     # How many DropOff locations are there at the same time
    max_dropoff_storage_size:        int  = 0     # How many items are needed until the dropoff is full
    max_agent_inventory_capacity:    int  = 5     # How many items are needed until the agent inventory is full


c = Constants
a = Actions


# noinspection PyAttributeOutsideInit, PyAbstractClass
class ItemFactory(BaseFactory):
    # noinspection PyMissingConstructor
    def __init__(self, *args, item_prop: ItemProperties = ItemProperties(), env_seed=time.time_ns(),
                 rewards_item: RewardsItem = RewardsItem(), **kwargs):
        if isinstance(item_prop, dict):
            item_prop = ItemProperties(**item_prop)
        if isinstance(rewards_item, dict):
            rewards_item = RewardsItem(**rewards_item)
        self.item_prop = item_prop
        self.rewards_item = rewards_item
        kwargs.update(env_seed=env_seed)
        self._item_rng = np.random.default_rng(env_seed)
        assert (item_prop.n_items <= ((1 + kwargs.get('_pomdp_r', 0) * 2) ** 2)) or not kwargs.get('_pomdp_r', 0)
        super().__init__(*args, **kwargs)

    @property
    def actions_hook(self) -> Union[Action, List[Action]]:
        # noinspection PyUnresolvedReferences
        super_actions = super().actions_hook
        super_actions.append(Action(str_ident=a.ITEM_ACTION))
        return super_actions

    @property
    def entities_hook(self) -> Dict[(str, Entities)]:
        # noinspection PyUnresolvedReferences
        super_entities = super().entities_hook

        empty_tiles = self[c.FLOOR].empty_tiles[:self.item_prop.n_drop_off_locations]
        drop_offs = DropOffLocations.from_tiles(
            empty_tiles, self._level_shape,
            entity_kwargs=dict(
                storage_size_until_full=self.item_prop.max_dropoff_storage_size)
        )
        item_register = ItemRegister(self._level_shape)
        empty_tiles = self[c.FLOOR].empty_tiles[:self.item_prop.n_items]
        item_register.spawn_items(empty_tiles)

        inventories = Inventories(self._obs_shape, self._level_shape)
        inventories.spawn_inventories(self[c.AGENT], self.item_prop.max_agent_inventory_capacity)

        super_entities.update({c.DROP_OFF: drop_offs, c.ITEM: item_register, c.INVENTORY: inventories})
        return super_entities

    def per_agent_raw_observations_hook(self, agent) -> Dict[str, np.typing.ArrayLike]:
        additional_raw_observations = super().per_agent_raw_observations_hook(agent)
        additional_raw_observations.update({c.INVENTORY: self[c.INVENTORY].by_entity(agent).as_array()})
        return additional_raw_observations

    def observations_hook(self) -> Dict[str, np.typing.ArrayLike]:
        additional_observations = super().observations_hook()
        additional_observations.update({c.ITEM: self[c.ITEM].as_array()})
        additional_observations.update({c.DROP_OFF: self[c.DROP_OFF].as_array()})
        return additional_observations

    def do_item_action(self, agent: Agent) -> (dict, dict):
        inventory = self[c.INVENTORY].by_entity(agent)
        if drop_off := self[c.DROP_OFF].by_pos(agent.pos):
            if inventory:
                valid = drop_off.place_item(inventory.pop())
            else:
                valid = c.NOT_VALID
            if valid:
                self.print(f'{agent.name} just dropped of an item at {drop_off.pos}.')
                info_dict = {f'{agent.name}_DROPOFF_VALID': 1, 'DROPOFF_VALID': 1}
            else:
                self.print(f'{agent.name} just tried to drop off at {agent.pos}, but failed.')
                info_dict = {f'{agent.name}_DROPOFF_FAIL': 1, 'DROPOFF_FAIL': 1}
            reward = dict(value=self.rewards_item.DROP_OFF_VALID if valid else self.rewards_item.DROP_OFF_FAIL,
                          reason=a.ITEM_ACTION, info=info_dict)
            return valid, reward
        elif item := self[c.ITEM].by_pos(agent.pos):
            item.change_register(inventory)
            item.set_tile_to(self._NO_POS_TILE)
            self.print(f'{agent.name} just picked up an item at {agent.pos}')
            info_dict = {f'{agent.name}_{a.ITEM_ACTION}_VALID': 1, f'{a.ITEM_ACTION}_VALID': 1}
            return c.VALID, dict(value=self.rewards_item.PICK_UP_VALID, reason=a.ITEM_ACTION, info=info_dict)
        else:
            self.print(f'{agent.name} just tried to pick up an item at {agent.pos}, but failed.')
            info_dict = {f'{agent.name}_{a.ITEM_ACTION}_FAIL': 1, f'{a.ITEM_ACTION}_FAIL': 1}
            return c.NOT_VALID, dict(value=self.rewards_item.PICK_UP_FAIL, reason=a.ITEM_ACTION, info=info_dict)

    def do_additional_actions(self, agent: Agent, action: Action) -> (dict, dict):
        # noinspection PyUnresolvedReferences
        action_result = super().do_additional_actions(agent, action)
        if action_result is None:
            if action == a.ITEM_ACTION:
                action_result = self.do_item_action(agent)
                return action_result
            else:
                return None
        else:
            return action_result

    def reset_hook(self) -> None:
        # noinspection PyUnresolvedReferences
        super().reset_hook()
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

    def step_hook(self) -> (List[dict], dict):
        # noinspection PyUnresolvedReferences
        super_reward_info = super().step_hook()
        for item in list(self[c.ITEM].values()):
            if item.auto_despawn >= 1:
                item.set_auto_despawn(item.auto_despawn-1)
            elif not item.auto_despawn:
                self[c.ITEM].delete_env_object(item)
            else:
                pass

        if not self._next_item_spawn:
            self.trigger_item_spawn()
        else:
            self._next_item_spawn = max(0, self._next_item_spawn-1)
        return super_reward_info

    def render_assets_hook(self, mode='human'):
        # noinspection PyUnresolvedReferences
        additional_assets = super().render_assets_hook()
        items = [RenderEntity(c.ITEM, item.tile.pos) for item in self[c.ITEM] if item.tile != self._NO_POS_TILE]
        additional_assets.extend(items)
        drop_offs = [RenderEntity(c.DROP_OFF, drop_off.tile.pos) for drop_off in self[c.DROP_OFF]]
        additional_assets.extend(drop_offs)
        return additional_assets


if __name__ == '__main__':
    from environments.utility_classes import AgentRenderOptions as aro, ObservationProperties

    render = True

    item_probs = ItemProperties(n_items=30, n_drop_off_locations=6)

    obs_props = ObservationProperties(render_agents=aro.SEPERATE, omit_agent_self=True, pomdp_r=2)

    move_props = {'allow_square_movement': True,
                  'allow_diagonal_movement': True,
                  'allow_no_op': False}

    factory = ItemFactory(n_agents=6, done_at_collision=False,
                          level_name='rooms', max_steps=400,
                          obs_prop=obs_props, parse_doors=True,
                          record_episodes=True, verbose=True,
                          mv_prop=move_props, item_prop=item_probs
                          )

    # noinspection DuplicatedCode
    n_actions = factory.action_space.n - 1
    obs_space = factory.observation_space
    obs_space_named = factory.named_observation_space

    for epoch in range(400):
        random_actions = [[random.randint(0, n_actions) for _
                           in range(factory.n_agents)] for _
                          in range(factory.max_steps + 1)]
        env_state = factory.reset()
        rwrd = 0
        for agent_i_action in random_actions:
            env_state, step_r, done_bool, info_obj = factory.step(agent_i_action)
            rwrd += step_r
            if render:
                factory.render()
            if done_bool:
                break
        print(f'Factory run {epoch} done, reward is:\n    {rwrd}')
pass
