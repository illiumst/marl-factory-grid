from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.clean_up.entitites import DirtPile
from marl_factory_grid.utils.results import Result


class DirtPiles(Collection):
    _entity = DirtPile

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
        return sum([dirt.amount for dirt in self])

    def __init__(self, *args, max_local_amount=5, clean_amount=1, max_global_amount: int = 20, coords_or_quantity=10,
                 initial_amount=2, amount_var=0.2, n_var=0.2, **kwargs):
        """
        A Collection of dirt piles that triggers their spawn.

        :param max_local_amount: The maximum amount of dirt allowed in a single pile at one position.
        :type max_local_amount: int

        :param clean_amount: The amount of dirt removed by a single cleaning action.
        :type clean_amount: int

        :param max_global_amount: The maximum total amount of dirt allowed in the environment.
        :type max_global_amount: int

        :param coords_or_quantity: Determines whether to use coordinates or quantity when triggering dirt pile spawn.
        :type coords_or_quantity:  Union[Tuple[int, int], int]

        :param initial_amount: The initial amount of dirt in each newly spawned pile.
        :type initial_amount: int

        :param amount_var: The variability in the initial amount of dirt in each pile.
        :type amount_var: float

        :param n_var: The variability in the number of new dirt piles spawned.
        :type n_var: float

        """
        super(DirtPiles, self).__init__(*args, **kwargs)
        self.amount_var = amount_var
        self.n_var = n_var
        self.clean_amount = clean_amount
        self.max_global_amount = max_global_amount
        self.max_local_amount = max_local_amount
        self.coords_or_quantity = coords_or_quantity
        self.initial_amount = initial_amount

    def trigger_spawn(self, state, coords_or_quantity=0, amount=0, ignore_blocking=False) -> [Result]:
        if ignore_blocking:
            print("##########################################")
            print("Blocking should not be ignored for this Entity")
            print("Exiting....")
            exit()
        coords_or_quantity = coords_or_quantity if coords_or_quantity else self.coords_or_quantity
        n_new = int(abs(coords_or_quantity + (state.rng.uniform(-self.n_var, self.n_var))))
        n_new = state.get_n_random_free_positions(n_new)

        amounts = [amount if amount else (self.initial_amount + state.rng.uniform(-self.amount_var, self.amount_var))
                   for _ in range(coords_or_quantity)]
        spawn_counter = 0
        for idx, (pos, a) in enumerate(zip(n_new, amounts)):
            if not self.global_amount > self.max_global_amount:
                if dirt := self.by_pos(pos):
                    dirt = next(dirt.iter())
                    new_value = dirt.amount + a
                    dirt.set_new_amount(new_value)
                else:
                    super().spawn([pos], amount=a)
                    spawn_counter += 1
            else:
                return Result(identifier=f'{self.name}_spawn', validity=c.NOT_VALID, value=spawn_counter)

        return Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=spawn_counter)

    def __repr__(self):
        s = super(DirtPiles, self).__repr__()
        return f'{s[:-1]}, {self.global_amount}]'
