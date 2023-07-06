from typing import List
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c
from . import rewards as r
from . import constants as M
from marl_factory_grid.utils.states import Gamestate


class MaintenanceRule(Rule):

    def __init__(self, n_maintainer: int = 1, *args, **kwargs):
        super(MaintenanceRule, self).__init__(*args, **kwargs)
        self.n_maintainer = n_maintainer

    def on_init(self, state: Gamestate, lvl_map):
        state[M.MAINTAINERS].spawn(state[c.FLOOR].empty_tiles[:self.n_maintainer], state)
        pass

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        for maintainer in state[M.MAINTAINERS]:
            maintainer.tick(state)
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        pass

    def on_check_done(self, state) -> List[DoneResult]:
        agents = list(state[c.AGENT].values())
        m_pos = state[M.MAINTAINERS].positions
        done_results = []
        for agent in agents:
            if agent.pos in m_pos:
                done_results.append(DoneResult(entity=agent, validity=c.VALID, identifier=self.name,
                                               reward=r.MAINTAINER_COLLISION_REWARD))
        return done_results
