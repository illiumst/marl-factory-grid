from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin
from marl_factory_grid.environment.entity.wall_floor import Floor
from marl_factory_grid.modules.clean_up.entitites import DirtPile

from marl_factory_grid.environment import constants as c


class DirtPiles(PositionMixin, EnvObjects):

    _entity = DirtPile
    is_blocking_light: bool = False
    can_collide: bool = False

    @property
    def amount(self):
        return sum([dirt.amount for dirt in self])

    def __init__(self, *args,
                 initial_amount=2,
                 initial_dirt_ratio=0.05,
                 dirt_spawn_r_var=0.1,
                 max_local_amount=5,
                 clean_amount=1,
                 max_global_amount: int = 20, **kwargs):
        super(DirtPiles, self).__init__(*args, **kwargs)
        self.clean_amount = clean_amount
        self.initial_amount = initial_amount
        self.initial_dirt_ratio = initial_dirt_ratio
        self.dirt_spawn_r_var = dirt_spawn_r_var
        self.max_global_amount = max_global_amount
        self.max_local_amount = max_local_amount

    def spawn(self, then_dirty_tiles, amount) -> bool:
        if isinstance(then_dirty_tiles, Floor):
            then_dirty_tiles = [then_dirty_tiles]
        for tile in then_dirty_tiles:
            if not self.amount > self.max_global_amount:
                if dirt := self.by_pos(tile.pos):
                    new_value = dirt.amount + amount
                    dirt.set_new_amount(new_value)
                else:
                    dirt = DirtPile(tile, initial_amount=amount, spawn_variation=self.dirt_spawn_r_var)
                    self.add_item(dirt)
            else:
                return c.NOT_VALID
        return c.VALID

    def trigger_dirt_spawn(self, state, initial_spawn=False) -> bool:
        free_for_dirt = [x for x in state[c.FLOOR]
                         if len(x.guests) == 0 or (
                                 len(x.guests) == 1 and
                                 isinstance(next(y for y in x.guests), DirtPile))
                         ]
        state.rng.shuffle(free_for_dirt)

        var = self.dirt_spawn_r_var
        new_spawn = abs(self.initial_dirt_ratio + (state.rng.uniform(-var, var) if initial_spawn else 0))
        n_dirt_tiles = max(0, int(new_spawn * len(free_for_dirt)))
        return self.spawn(free_for_dirt[:n_dirt_tiles], self.initial_amount)

    def __repr__(self):
        s = super(DirtPiles, self).__repr__()
        return f'{s[:-1]}, {self.amount})'
