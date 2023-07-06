from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.helpers import is_move
from marl_factory_grid.utils.results import TickResult

from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.clean_up import constants as d


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
                        if state[d.DIRT].spawn(entity.tile, amount=smeared_dirt):
                            results.append(TickResult(identifier=self.name, entity=entity,
                                                      reward=0, validity=c.VALID))
        return results
