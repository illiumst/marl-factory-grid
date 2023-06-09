from typing import List
from environment.rules import Rule
from environment.utils.results import TickResult, DoneResult


class TemplateRule(Rule):

    def __init__(self, *args, **kwargs):
        super(TemplateRule, self).__init__(*args, **kwargs)

    def on_init(self, state):
        pass

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        pass

    def tick_post_step(self, state) -> List[TickResult]:
        pass

    def on_check_done(self, state) -> List[DoneResult]:
        pass
