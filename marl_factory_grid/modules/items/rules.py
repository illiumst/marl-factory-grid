from typing import List

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.modules.items import constants as i


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
        state[i.DROP_OFF].trigger_drop_off_location_spawn(state, self.n_locations)
        self._next_item_spawn = self.spawn_frequency
        state[i.INVENTORY].trigger_inventory_spawn(state)
        state[i.ITEM].trigger_item_spawn(state, self.n_items, self.spawn_frequency)

    def tick_step(self, state):
        for item in list(state[i.ITEM].values()):
            if item.auto_despawn >= 1:
                item.set_auto_despawn(item.auto_despawn - 1)
            elif not item.auto_despawn:
                state[i.ITEM].delete_env_object(item)
            else:
                pass

        if not self._next_item_spawn:
            state[i.ITEM].trigger_item_spawn(state, self.n_items, self.spawn_frequency)
        else:
            self._next_item_spawn = max(0, self._next_item_spawn - 1)
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        for item in list(state[i.ITEM].values()):
            if item.auto_despawn >= 1:
                item.set_auto_despawn(item.auto_despawn-1)
            elif not item.auto_despawn:
                state[i.ITEM].delete_env_object(item)
            else:
                pass

        if not self._next_item_spawn:
            if spawned_items := state[i.ITEM].trigger_item_spawn(state, self.n_items, self.spawn_frequency):
                return [TickResult(self.name, validity=c.VALID, value=spawned_items, entity=None)]
            else:
                return [TickResult(self.name, validity=c.NOT_VALID, value=0, entity=None)]
        else:
            self._next_item_spawn = max(0, self._next_item_spawn-1)
            return []

