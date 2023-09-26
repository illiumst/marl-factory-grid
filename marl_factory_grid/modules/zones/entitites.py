import random
from typing import List, Tuple

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.entity.object import Object
from marl_factory_grid.environment.entity.wall_floor import Floor
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.doors import constants as d


class Zone(Object):

    @property
    def positions(self):
        return self.coords

    def __init__(self, coords: List[Tuple[(int, int)]], *args, **kwargs):
        super(Zone, self).__init__(*args, **kwargs)
        self.coords = coords

    @property
    def random_tile(self):
        return random.choice(self.coords)
