from marl_factory_grid.modules.clean_up import constants as d, rewards as r
from marl_factory_grid.environment import constants as c

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.helpers import is_move
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.utils.results import DoneResult


class DirtAllCleanDone(Rule):

    def __init__(self):
        super().__init__()

    def on_check_done(self, state) -> [DoneResult]:
        if len(state[d.DIRT]) == 0 and state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name, reward=r.CLEAN_UP_ALL)]
        return [DoneResult(validity=c.NOT_VALID, identifier=self.name, reward=0)]


class DirtRespawnRule(Rule):

    def __init__(self, spawn_freq=15):
        super().__init__()
        self.spawn_freq = spawn_freq
        self._next_dirt_spawn = spawn_freq

    def on_init(self, state, lvl_map) -> str:
        state[d.DIRT].trigger_dirt_spawn(state, initial_spawn=True)
        return f'Initial Dirt was spawned on: {[x.pos for x in state[d.DIRT]]}'

    def tick_step(self, state):
        if self._next_dirt_spawn < 0:
            pass  # No DirtPile Spawn
        elif not self._next_dirt_spawn:
            validity = state[d.DIRT].trigger_dirt_spawn(state)

            return [TickResult(entity=None, validity=validity, identifier=self.name, reward=0)]
            self._next_dirt_spawn = self.spawn_freq
        else:
            self._next_dirt_spawn -= 1
        return []


class DirtSmearOnMove(Rule):

    def __init__(self, smear_amount: float = 0.2):
        super().__init__()
        self.smear_amount = smear_amount

    def tick_post_step(self, state):
        results = list()
        for entity in state.moving_entites:
            if is_move(entity.state.identifier) and entity.state.validity == c.VALID:
                if old_pos_dirt := state[d.DIRT].by_pos(entity.last_pos):
                    if smeared_dirt := round(old_pos_dirt.amount * self.smear_amount, 2):
                        if state[d.DIRT].spawn(entity.pos, amount=smeared_dirt):
                            results.append(TickResult(identifier=self.name, entity=entity,
                                                      reward=0, validity=c.VALID))
        return results
