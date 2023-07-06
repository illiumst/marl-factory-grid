from typing import List

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.modules.items import constants as i
from marl_factory_grid.modules.items.entitites import DropOffLocation


class ItemRules(Rule):

    def __init__(self, n_items: int = 5, spawn_frequency: int = 15,
                 n_locations: int = 5, max_dropoff_storage_size: int = 0):
        super().__init__()
        self.spawn_frequency = spawn_frequency
        self._next_item_spawn = spawn_frequency
        self.n_items = n_items
        self.max_dropoff_storage_size = max_dropoff_storage_size
        self.n_locations = n_locations

    def on_init(self, state, lvl_map):
        self.trigger_drop_off_location_spawn(state)
        self._next_item_spawn = self.spawn_frequency
        self.trigger_inventory_spawn(state)
        self.trigger_item_spawn(state)

    def tick_step(self, state):
        for item in list(state[i.ITEM].values()):
            if item.auto_despawn >= 1:
                item.set_auto_despawn(item.auto_despawn - 1)
            elif not item.auto_despawn:
                state[i.ITEM].delete_env_object(item)
            else:
                pass

        if not self._next_item_spawn:
            self.trigger_item_spawn(state)
        else:
            self._next_item_spawn = max(0, self._next_item_spawn - 1)
        return []

    def trigger_item_spawn(self, state):
        if item_to_spawns := max(0, (self.n_items - len(state[i.ITEM]))):
            empty_tiles = state[c.FLOOR].empty_tiles[:item_to_spawns]
            state[i.ITEM].spawn(empty_tiles)
            self._next_item_spawn = self.spawn_frequency
            state.print(f'{item_to_spawns} new items have been spawned; next spawn in {self._next_item_spawn}')
            return len(empty_tiles)
        else:
            state.print('No Items are spawning, limit is reached.')
            return 0

    @staticmethod
    def trigger_inventory_spawn(state):
        state[i.INVENTORY].spawn(state[c.AGENT])

    def tick_post_step(self, state) -> List[TickResult]:
        for item in list(state[i.ITEM].values()):
            if item.auto_despawn >= 1:
                item.set_auto_despawn(item.auto_despawn-1)
            elif not item.auto_despawn:
                state[i.ITEM].delete_env_object(item)
            else:
                pass

        if not self._next_item_spawn:
            if spawned_items := self.trigger_item_spawn(state):
                return [TickResult(self.name, validity=c.VALID, value=spawned_items, entity=None)]
            else:
                return [TickResult(self.name, validity=c.NOT_VALID, value=0, entity=None)]
        else:
            self._next_item_spawn = max(0, self._next_item_spawn-1)
            return []

    def trigger_drop_off_location_spawn(self, state):
        empty_tiles = state[c.FLOOR].empty_tiles[:self.n_locations]
        do_entites = state[i.DROP_OFF]
        drop_offs = [DropOffLocation(tile) for tile in empty_tiles]
        do_entites.add_items(drop_offs)
