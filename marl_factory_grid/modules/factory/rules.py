import random
from typing import List, Union

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult


class AgentSingleZonePlacementBeta(Rule):

    def __init__(self):
        super().__init__()

    def on_init(self, state, lvl_map):
        zones = state[c.ZONES]
        n_zones = state[c.ZONES]
        agents = state[c.AGENT]
        if len(self.coordinates) == len(agents):
            coordinates = self.coordinates
        elif len(self.coordinates) > len(agents):
            coordinates = random.choices(self.coordinates, k=len(agents))
        else:
            raise ValueError
        tiles = [state[c.FLOOR].by_pos(pos) for pos in coordinates]
        for agent, tile in zip(agents, tiles):
            agent.move(tile)

    def tick_step(self, state):
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        return []