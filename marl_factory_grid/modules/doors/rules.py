from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult
from . import constants as d
from .entitites import DoorIndicator


class DoorAutoClose(Rule):

    def __init__(self, close_frequency: int = 10):
        """
        This rule closes doors, that have been opened automatically, when no entity is blocking the position.

        :type close_frequency: int
        :param close_frequency: How many ticks after opening, should the door close?
        """
        super().__init__()
        self.close_frequency = close_frequency

    def tick_step(self, state):
        if doors := state[d.DOORS]:
            doors_tick_result = doors.tick_doors(state)
            doors_that_ticked = [key for key, val in doors_tick_result.items() if val]
            state.print(f'{doors_that_ticked} were auto-closed'
                        if doors_that_ticked else 'No Doors were auto-closed')
            return [TickResult(self.name, validity=c.VALID, value=1)]
        state.print('There are no doors, but you loaded the corresponding Module')
        return []


class IndicateDoorAreaInObservation(Rule):

    def __init__(self):
        """
        Whether to add an indicator aka place a value around doors (within the door tile), for agents to see.
        This could improve learning.

        """
        # Todo: Discuss if this rather be a single entity with a standalone observation.
        #  Could then be combined with the "Combine"-approach.
        super().__init__()

    def on_init(self, state, lvl_map):
        for door in state[d.DOORS]:
            state[d.DOORS].add_items([DoorIndicator(x) for x in state.entities.neighboring_positions(door.pos)])
