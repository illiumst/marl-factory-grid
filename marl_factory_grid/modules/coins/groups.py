import ast
import random

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.coins.entitites import CoinPile
from marl_factory_grid.utils.results import Result
from marl_factory_grid.utils import helpers as h


class CoinPiles(Collection):
    _entity = CoinPile

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_move(self):
        return False

    @property
    def var_has_position(self):
        return True

    @property
    def global_amount(self) -> float:
        """
        Internal Usage
        """
        return sum([coin.amount for coin in self])

    def __init__(self, *args, max_local_amount=5, collect_amount=1, max_global_amount: int = 20, coords_or_quantity=10,
                 initial_amount=2, amount_var=0.2, n_var=0.2, randomize=False, randomization_seed=0, **kwargs):
        """
        A Collection of coin piles that triggers their spawn.

        :param max_local_amount: The maximum amount of coins allowed in a single pile at one position.
        :type max_local_amount: int

        :param clean_amount: The amount of coins removed by a single collecting action.
        :type clean_amount: int

        :param max_global_amount: The maximum total amount of coins allowed in the environment.
        :type max_global_amount: int

        :param coords_or_quantity: Determines whether to use coordinates or quantity when triggering coin pile spawn.
        :type coords_or_quantity:  Union[Tuple[int, int], int]

        :param initial_amount: The initial amount of coin in each newly spawned pile.
        :type initial_amount: int

        :param amount_var: The variability in the initial amount of coin in each pile.
        :type amount_var: float

        :param n_var: The variability in the number of new coin piles spawned.
        :type n_var: float

        """
        super(CoinPiles, self).__init__(*args, **kwargs)
        self.amount_var = amount_var
        self.n_var = n_var
        self.collect_amount = collect_amount
        self.max_global_amount = max_global_amount
        self.max_local_amount = max_local_amount
        self.coords_or_quantity = coords_or_quantity
        self.initial_amount = initial_amount
        self.randomize = randomize
        self.randomized_selection = None

    def trigger_spawn(self, state, coords_or_quantity=0, amount=0, ignore_blocking=False) -> [Result]:
        if ignore_blocking:
            print("##########################################")
            print("Blocking should not be ignored for this Entity")
            print("Exiting....")
            exit()
        coords_or_quantity = coords_or_quantity if coords_or_quantity else self.coords_or_quantity
        if isinstance(coords_or_quantity, int):
            n_new = int(abs(coords_or_quantity + (state.rng.uniform(-self.n_var, self.n_var))))
            n_new = state.get_n_random_free_positions(n_new)
        else:
            coords_or_quantity = ast.literal_eval(coords_or_quantity)
            if isinstance(coords_or_quantity[0], int):
                n_new = [coords_or_quantity]
            else:
                n_new = [pos for pos in coords_or_quantity]

        if self.randomize:
            if not self.randomized_selection:
                n_new_prime = []
                for n in n_new:
                    if random.random() < 0.5:
                        n_new_prime.append(n)
                n_new = n_new_prime
                self.randomized_selection = n_new
            else:
                n_new = self.randomized_selection
        amounts = [amount if amount else (self.initial_amount)  # removed rng amount
                   for _ in range(len(n_new))]

        spawn_counter = 0
        for idx, (pos, a) in enumerate(zip(n_new, amounts)):
            if not self.global_amount > self.max_global_amount:
                if coin := self.by_pos(pos):
                    coin = h.get_first(coin)
                    new_value = coin.amount + a
                    coin.set_new_amount(new_value)
                else:
                    super().spawn([pos], amount=a)
                    spawn_counter += 1
            else:
                return Result(identifier=f'{self.name}_spawn', validity=c.NOT_VALID, value=spawn_counter)

        return Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=spawn_counter)

    def __repr__(self):
        s = super(CoinPiles, self).__repr__()
        return f'{s[:-1]}, {self.global_amount}]'
