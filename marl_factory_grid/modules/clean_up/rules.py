from marl_factory_grid.modules.clean_up import constants as d
from marl_factory_grid.environment import constants as c

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.helpers import is_move
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.utils.results import DoneResult


class DoneOnAllDirtCleaned(Rule):

    def __init__(self, reward: float = d.REWARD_CLEAN_UP_ALL):
        """
        Defines a 'Done'-condition which triggers, when there is no more 'Dirt' in the environment.

        :type reward: float
        :parameter reward: Given reward when condition triggers.
        """
        super().__init__()
        self.reward = reward

    def on_check_done(self, state) -> [DoneResult]:
        if len(state[d.DIRT]) == 0 and state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name, reward=self.reward)]
        return []


class RespawnDirt(Rule):

    def __init__(self, respawn_freq: int = 15, respawn_n: int = 5, respawn_amount: float = 1.0):
        """
        Defines the spawn pattern of initial and additional 'Dirt'-entities.
        First chooses positions, then tries to spawn dirt until 'respawn_n' or the maximal global amount is reached.
        If there is already some, it is topped up to min(max_local_amount, amount).

        :type respawn_freq: int
        :parameter respawn_freq: In which frequency should this Rule try to spawn new 'Dirt'?
        :type respawn_n: int
        :parameter respawn_n: How many respawn positions are considered.
        :type respawn_amount: float
        :parameter respawn_amount: Defines how much dirt 'amount' is placed every 'spawn_freq' ticks.
        """
        super().__init__()
        self.respawn_n = respawn_n
        self.respawn_amount = respawn_amount
        self.respawn_freq = respawn_freq
        self._next_dirt_spawn = respawn_freq

    def tick_step(self, state):
        collection = state[d.DIRT]
        if self._next_dirt_spawn < 0:
            result = []  # No DirtPile Spawn
        elif not self._next_dirt_spawn:
            result = [collection.trigger_spawn(state, coords_or_quantity=self.respawn_n, amount=self.respawn_amount)]
            self._next_dirt_spawn = self.respawn_freq
        else:
            self._next_dirt_spawn -= 1
            result = []
        return result


class EntitiesSmearDirtOnMove(Rule):

    def __init__(self, smear_ratio: float = 0.2):
        """
        Enables 'smearing'. Entities that move through dirt, will leave a trail behind them.
        They take dirt * smear_ratio of it with them to their next position.

        :type smear_ratio: float
        :parameter smear_ratio: How much percent dirt is smeared by entities to their next position.
        """
        assert smear_ratio < 1, "'Smear Amount' must be smaller than 1"
        super().__init__()
        self.smear_ratio = smear_ratio

    def tick_post_step(self, state):
        results = list()
        for entity in state.moving_entites:
            if is_move(entity.state.identifier) and entity.state.validity == c.VALID:
                if old_pos_dirt := state[d.DIRT].by_pos(entity.last_pos):
                    old_pos_dirt = next(iter(old_pos_dirt))
                    if smeared_dirt := round(old_pos_dirt.amount * self.smear_ratio, 2):
                        if state[d.DIRT].spawn(entity.pos, amount=smeared_dirt):
                            results.append(TickResult(identifier=self.name, entity=entity,
                                                      validity=c.VALID, value=smeared_dirt))
        return results
