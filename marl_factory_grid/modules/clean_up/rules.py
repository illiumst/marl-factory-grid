from marl_factory_grid.modules.clean_up import constants as d, rewards as r
from marl_factory_grid.environment import constants as c

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.helpers import is_move
from marl_factory_grid.utils.results import TickResult
from marl_factory_grid.utils.results import DoneResult


class DoneOnAllDirtCleaned(Rule):

    def __init__(self, reward: float = r.CLEAN_UP_ALL):
        """
        Defines a 'Done'-condition which tirggers, when there is no more 'Dirt' in the environment.

        :type reward: float
        :parameter reward: Given reward when condition triggers.
        """
        super().__init__()
        self.reward = reward

    def on_check_done(self, state) -> [DoneResult]:
        if len(state[d.DIRT]) == 0 and state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name, reward=self.reward)]
        return [DoneResult(validity=c.NOT_VALID, identifier=self.name, reward=0)]


class SpawnDirt(Rule):

    def __init__(self, initial_n: int = 5, initial_amount: float = 1.3,
                 respawn_n: int = 3, respawn_amount: float = 0.8,
                 n_var: float = 0.2, amount_var: float = 0.2, spawn_freq: int = 15):
        """
        Defines the spawn pattern of intial and additional 'Dirt'-entitites.
        First chooses positions, then trys to spawn dirt until 'respawn_n' or the maximal global amount is reached.
        If there is allready some, it is topped up to min(max_local_amount, amount).

        :type spawn_freq: int
        :parameter spawn_freq: In which frequency should this Rule try to spawn new 'Dirt'?
        :type respawn_n: int
        :parameter respawn_n: How many respawn positions are considered.
        :type initial_n: int
        :parameter initial_n: How much initial positions are considered.
        :type amount_var: float
        :parameter amount_var: Variance of amount to spawn.
        :type n_var: float
        :parameter n_var: Variance of n to spawn.
        :type respawn_amount: float
        :parameter respawn_amount: Defines how much dirt 'amount' is placed every 'spawn_freq' ticks.
        :type initial_amount: float
        :parameter initial_amount: Defines how much dirt 'amount' is initially placed.

        """
        super().__init__()
        self.amount_var = amount_var
        self.n_var = n_var
        self.respawn_amount = respawn_amount
        self.respawn_n = respawn_n
        self.initial_amount = initial_amount
        self.initial_n = initial_n
        self.spawn_freq = spawn_freq
        self._next_dirt_spawn = spawn_freq

    def on_init(self, state, lvl_map) -> str:
        result = state[d.DIRT].trigger_dirt_spawn(self.initial_n, self.initial_amount, state,
                                                  n_var=self.n_var, amount_var=self.amount_var)
        state.print(f'Initial Dirt was spawned on: {[x.pos for x in state[d.DIRT]]}')
        return result

    def tick_step(self, state):
        if self._next_dirt_spawn < 0:
            pass  # No DirtPile Spawn
        elif not self._next_dirt_spawn:
            result = [state[d.DIRT].trigger_dirt_spawn(self.respawn_n, self.respawn_amount, state,
                                                       n_var=self.n_var, amount_var=self.amount_var)]
            self._next_dirt_spawn = self.spawn_freq
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
                    if smeared_dirt := round(old_pos_dirt.amount * self.smear_ratio, 2):
                        if state[d.DIRT].spawn(entity.pos, amount=smeared_dirt):
                            results.append(TickResult(identifier=self.name, entity=entity,
                                                      reward=0, validity=c.VALID))
        return results
