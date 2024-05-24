import unittest
from typing import List
from marl_factory_grid.utils.results import TickResult, DoneResult


class Test(unittest.TestCase):

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self):
        """
        Base test class for unit tests that provides base functions to be overwritten that are automatically called by
        the StepTests class.
        """
        super().__init__()

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
