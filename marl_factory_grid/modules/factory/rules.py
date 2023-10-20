import random
from typing import List, Union

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult


class AgentSingleZonePlacementBeta(Rule):

    def __init__(self):
        raise NotImplementedError()
        # TODO!!!! Is this concept needed any more?
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

        for agent, pos in zip(agents, coordinates):
            agent.move(pos, state)

    def tick_step(self, state):
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        return []