from typing import List

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.modules.items import constants as i


class RespawnItems(Rule):

    def __init__(self, n_items: int = 5, respawn_freq: int = 15, n_locations: int = 5):
        """
        Defines the respawning behaviour of items.

        :param n_items: Specifies how many items should respawn.
        :type n_items: int
        :param respawn_freq: Specifies how often items should respawn.
        :type respawn_freq: int
        :param n_locations: Specifies at how many locations items should be able to respawn.
        :type: int
        """
        super().__init__()
        self.spawn_frequency = respawn_freq
        self._next_item_spawn = respawn_freq
        self.n_items = n_items
        self.n_locations = n_locations

    def tick_step(self, state):
        if not self._next_item_spawn:
            state[i.ITEM].trigger_spawn(state, self.n_items, self.spawn_frequency)
        else:
            self._next_item_spawn = max(0, self._next_item_spawn - 1)
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        if not self._next_item_spawn:
            if spawned_items := state[i.ITEM].trigger_spawn(state, self.n_items, self.spawn_frequency):
                return [TickResult(self.name, validity=c.VALID, value=spawned_items.value)]
            else:
                return [TickResult(self.name, validity=c.NOT_VALID, value=0)]
        else:
            self._next_item_spawn = max(0, self._next_item_spawn-1)
            return []

