from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.wall import Wall
from marl_factory_grid.environment.groups.collection import Collection


class Walls(Collection):
    _entity = Wall
    symbol = c.SYMBOL_WALL

    var_can_collide = True
    var_is_blocking_light = True
    var_can_move = False
    var_has_position = True
    var_can_be_bound = False
    var_is_blocking_pos = True

    def __init__(self, *args, **kwargs):
        super(Walls, self).__init__(*args, **kwargs)
        self._value = c.VALUE_OCCUPIED_CELL

    def by_pos(self, pos: (int, int)):
        try:
            return super().by_pos(pos)[0]
        except IndexError:
            return None

    def reset(self):
        pass

