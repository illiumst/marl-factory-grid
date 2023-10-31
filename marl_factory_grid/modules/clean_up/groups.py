from typing import Union, List, Tuple

from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import Result
from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.clean_up.entitites import DirtPile


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
    def amount(self):
        return sum([dirt.amount for dirt in self])

    def __init__(self, *args,
                 max_local_amount=5,
                 clean_amount=1,
                 max_global_amount: int = 20, **kwargs):
        super(DirtPiles, self).__init__(*args, **kwargs)
        self.clean_amount = clean_amount
        self.max_global_amount = max_global_amount
        self.max_local_amount = max_local_amount

    def spawn(self, coords_or_quantity: Union[int, List[Tuple[(int, int)]]], *entity_args):
        amount_s = entity_args[0]
        spawn_counter = 0
        for idx, pos in enumerate(coords_or_quantity):
            if not self.amount > self.max_global_amount:
                amount = amount_s[idx] if isinstance(amount_s, list) else amount_s
                if dirt := self.by_pos(pos):
                    dirt = next(dirt.iter())
                    new_value = dirt.amount + amount
                    dirt.set_new_amount(new_value)
                else:
                    dirt = DirtPile(pos, amount=amount)
                    self.add_item(dirt)
                    spawn_counter += 1
            else:
                return Result(identifier=f'{self.name}_spawn', validity=c.NOT_VALID, reward=0,
                              value=spawn_counter)
        return Result(identifier=f'{self.name}_spawn', validity=c.VALID, reward=0, value=spawn_counter)

    def trigger_dirt_spawn(self, n, amount, state, n_var=0.2, amount_var=0.2) -> Result:
        free_for_dirt = [x for x in state.entities.floorlist if len(state.entities.pos_dict[x]) == 0 or (
                len(state.entities.pos_dict[x]) >= 1 and isinstance(next(y for y in x), DirtPile))]
        # free_for_dirt = [x for x in state[c.FLOOR]
        #                  if len(x.guests) == 0 or (
        #                          len(x.guests) == 1 and
        #                          isinstance(next(y for y in x.guests), DirtPile))]
        state.rng.shuffle(free_for_dirt)

        new_spawn = int(abs(n + (state.rng.uniform(-n_var, n_var))))
        new_amount_s = [abs(amount + (amount*state.rng.uniform(-amount_var, amount_var))) for _ in range(new_spawn)]
        n_dirty_positions = free_for_dirt[:new_spawn]
        return self.spawn(n_dirty_positions, new_amount_s)

    def __repr__(self):
        s = super(DirtPiles, self).__repr__()
        return f'{s[:-1]}, {self.amount})'
