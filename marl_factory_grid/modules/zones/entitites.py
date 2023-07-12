import random
from typing import List

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.entity.object import Object
from marl_factory_grid.environment.entity.wall_floor import Floor
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.doors import constants as d


class Zone(Object):

    @property
    def positions(self):
        return [x.pos for x in self.tiles]

    def __init__(self, tiles: List[Floor], *args, **kwargs):
        super(Zone, self).__init__(*args, **kwargs)
        self.tiles = tiles

    @property
    def random_tile(self):
        return random.choice(self.tiles)
