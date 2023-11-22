from typing import List

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c
from . import constants as M


class MoveMaintainers(Rule):

    def __init__(self):
        """
        This rule is responsible for moving the maintainers at every step of the environment.
        """
        super().__init__()

    def tick_step(self, state) -> List[TickResult]:
        for maintainer in state[M.MAINTAINERS]:
            maintainer.tick(state)
        # Todo: Return a Result Object.
        return []


class DoneAtMaintainerCollision(Rule):

    def __init__(self):
        """
        When active, this rule stops the environment after a maintainer reports a collision with another entity.
        """
        super().__init__()

    def on_check_done(self, state) -> List[DoneResult]:
        agents = list(state[c.AGENT].values())
        m_pos = state[M.MAINTAINERS].positions
        done_results = []
        for agent in agents:
            if agent.pos in m_pos:
                done_results.append(DoneResult(entity=agent, validity=c.VALID, identifier=self.name,
                                               reward=M.MAINTAINER_COLLISION_REWARD))
        return done_results
