from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult

from marl_factory_grid.modules.clean_up import constants as d


class DirtRespawnRule(Rule):

    def __init__(self, spawn_freq=15):
        super().__init__()
        self.spawn_freq = spawn_freq
        self._next_dirt_spawn = spawn_freq

    def on_init(self, state, lvl_map) -> str:
        state[d.DIRT].trigger_dirt_spawn(state, initial_spawn=True)
        return f'Initial Dirt was spawned on: {[x.pos for x in state[d.DIRT]]}'

    def tick_step(self, state):
        if self._next_dirt_spawn < 0:
            pass  # No DirtPile Spawn
        elif not self._next_dirt_spawn:
            validity = state[d.DIRT].trigger_dirt_spawn(state)

            return [TickResult(entity=None, validity=validity, identifier=self.name, reward=0)]
            self._next_dirt_spawn = self.spawn_freq
        else:
            self._next_dirt_spawn -= 1
        return []
