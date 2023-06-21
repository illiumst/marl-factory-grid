from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.modules.doors import constants as d


class DoorAutoClose(Rule):

    def __init__(self, close_frequency: int = 10):
        super().__init__()
        self.close_frequency = close_frequency

    def tick_step(self, state):
        if doors := state[d.DOORS]:
            doors_tick_result = doors.tick_doors()
            doors_that_ticked = [key for key, val in doors_tick_result.items() if val]
            state.print(f'{doors_that_ticked} were auto-closed'
                        if doors_that_ticked else 'No Doors were auto-closed')
            return [TickResult(self.name, validity=c.VALID, value=0)]
        state.print('There are no doors, but you loaded the corresponding Module')
        return []
