import abc
from typing import List

from marl_factory_grid.utils.results import TickResult, DoneResult


class Test(abc.ABC):

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self):
        pass

    def __repr__(self):
        return f'{self.name}'

    def on_init(self, state, lvl_map):
        return []

    def on_reset(self):
        return []

    def tick_pre_step(self, state) -> List[TickResult]:
        return []

    def tick_step(self, state) -> List[TickResult]:
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        return []


class FirstTest(Test):

    def __init__(self):
        print("firstTest")
        super().__init__()
        pass
