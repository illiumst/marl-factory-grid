from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import DoneResult
from marl_factory_grid.modules.clean_up import constants as d, rewards as r


class DirtAllCleanDone(Rule):

    def __init__(self):
        super().__init__()

    def on_check_done(self, state) -> [DoneResult]:
        if len(state[d.DIRT]) == 0 and state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name, reward=r.CLEAN_UP_ALL)]
        return [DoneResult(validity=c.NOT_VALID, identifier=self.name, reward=0)]
