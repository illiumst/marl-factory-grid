from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.wall import Wall
from marl_factory_grid.environment.groups.collection import Collection


class Walls(Collection):
    _entity = Wall
    symbol = c.SYMBOL_WALL

    @property
    def var_has_position(self):
        return True

    def __init__(self, *args, **kwargs):
        super(Walls, self).__init__(*args, **kwargs)
        self._value = c.VALUE_OCCUPIED_CELL

    #ToDo: Do we need this? Move to spawn methode?
    # @classmethod
    # def from_coordinates(cls, argwhere_coordinates, *args, **kwargs):
    #     tiles = cls(*args, **kwargs)
    #     # noinspection PyTypeChecker
    #     tiles.add_items([cls._entity(pos) for pos in argwhere_coordinates])
    #     return tiles

    def by_pos(self, pos: (int, int)):
        try:
            return super().by_pos(pos)[0]
        except IndexError:
            return None
