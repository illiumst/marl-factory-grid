from marl_factory_grid.modules.coins import constants as d
from marl_factory_grid.environment import constants as c

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.helpers import is_move
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.utils.results import DoneResult


class DoneOnAllCoinsCollected(Rule):

    def __init__(self, reward: float = d.REWARD_COLLECT_ALL):
        """
        Defines a 'Done'-condition which triggers, when there is no more 'Dirt' in the environment.

        :type reward: float
        :parameter reward: Given reward when condition triggers.
        """
        super().__init__()
        self.reward = reward

    def on_check_done(self, state) -> [DoneResult]:
        if len(state[d.COIN]) == 0 and state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name, reward=self.reward)]
        return []


class RespawnCoins(Rule):

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
        self._next_coin_spawn = respawn_freq

    def tick_step(self, state):
        collection = state[d.COIN]
        if self._next_coin_spawn < 0:
            result = []  # No CoinPile Spawn
        elif not self._next_coin_spawn:
            result = [collection.trigger_spawn(state, coords_or_quantity=self.respawn_n, amount=self.respawn_amount)]
            self._next_coin_spawn = self.respawn_freq
        else:
            self._next_coin_spawn -= 1
            result = []
        return result
