from typing import List
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult


class TemplateRule(Rule):

    def __init__(self, *args, **kwargs):
        super(TemplateRule, self).__init__(*args, **kwargs)

    def on_init(self, state, lvl_map):
        pass

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        pass

    def tick_post_step(self, state) -> List[TickResult]:
        pass

    def on_check_done(self, state) -> List[DoneResult]:
        pass
