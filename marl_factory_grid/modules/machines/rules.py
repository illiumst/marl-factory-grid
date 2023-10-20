from typing import List
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.machines import constants as m
from marl_factory_grid.modules.machines.entitites import Machine


class MachineRule(Rule):

    def __init__(self, n_machines: int = 2):
        super(MachineRule, self).__init__()
        self.n_machines = n_machines

    def on_init(self, state, lvl_map):
        # TODO Move to spawn!!!
        state[m.MACHINES].add_items(Machine(pos) for pos in state.entities.empty_positions())

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        pass

    def tick_post_step(self, state) -> List[TickResult]:
        pass

    def on_check_done(self, state) -> List[DoneResult]:
        pass


class DoneOnBreakRule(Rule):

    def on_check_done(self, state) -> List[DoneResult]:
        pass