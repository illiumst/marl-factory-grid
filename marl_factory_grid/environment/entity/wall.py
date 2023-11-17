from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.utility_classes import RenderEntity


class Wall(Entity):

    def __init__(self, *args, **kwargs):
        """
        TODO


        :return:
        """
        super().__init__(*args, **kwargs)

    @property
    def encoding(self):
        return c.VALUE_OCCUPIED_CELL

    def render(self):
        return RenderEntity(c.WALL, self.pos)
