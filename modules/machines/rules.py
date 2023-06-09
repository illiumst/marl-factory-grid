from typing import List
from environment.rules import Rule
from environment.utils.results import TickResult, DoneResult
from environment import constants as c
from modules.machines import constants as m
from modules.machines.entitites import Machine


class MachineRule(Rule):

    def __init__(self, n_machines: int = 2):
        super(MachineRule, self).__init__()
        self.n_machines = n_machines

    def on_init(self, state):
        empty_tiles = state[c.FLOOR].empty_tiles[:self.n_machines]
        state[m.MACHINES].add_items(Machine(tile) for tile in empty_tiles)

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        pass

    def tick_post_step(self, state) -> List[TickResult]:
        pass

    def on_check_done(self, state) -> List[DoneResult]:
        pass
