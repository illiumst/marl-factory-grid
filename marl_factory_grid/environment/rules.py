import abc
from typing import List

from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import rewards as r, constants as c


class Rule(abc.ABC):

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


class MaxStepsReached(Rule):

    def __init__(self, max_steps: int = 500):
        super().__init__()
        self.max_steps = max_steps

    def on_init(self, state, lvl_map):
        pass

    def on_check_done(self, state):
        if self.max_steps <= state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name, reward=0)]
        return [DoneResult(validity=c.NOT_VALID, identifier=self.name, reward=0)]


class AssignGlobalPositions(Rule):

    def __init__(self):
        super().__init__()

    def on_init(self, state, lvl_map):
        from marl_factory_grid.environment.entity.util import GlobalPosition
        for agent in state[c.AGENT]:
            gp = GlobalPosition(lvl_map.level_shape)
            gp.bind_to(agent)
            state[c.GLOBALPOSITIONS].add_item(gp)
        return []


class Collision(Rule):

    def __init__(self, done_at_collisions: bool = False):
        super().__init__()
        self.done_at_collisions = done_at_collisions
        self.curr_done = False

    def tick_post_step(self, state) -> List[TickResult]:
        self.curr_done = False
        tiles_with_collisions = state.get_all_tiles_with_collisions()
        results = list()
        for tile in tiles_with_collisions:
            guests = tile.guests_that_can_collide
            if len(guests) >= 2:
                for i, guest in enumerate(guests):
                    try:
                        guest.set_state(TickResult(identifier=c.COLLISION, reward=r.COLLISION,
                                                   validity=c.NOT_VALID, entity=self))
                    except AttributeError:
                        pass
                    results.append(TickResult(entity=guest, identifier=c.COLLISION,
                                              reward=r.COLLISION, validity=c.VALID))
                self.curr_done = True
        return results

    def on_check_done(self, state) -> List[DoneResult]:
        if self.curr_done and self.done_at_collisions:
            return [DoneResult(validity=c.VALID, identifier=c.COLLISION, reward=r.COLLISION)]
        return [DoneResult(validity=c.NOT_VALID, identifier=self.name, reward=0)]
