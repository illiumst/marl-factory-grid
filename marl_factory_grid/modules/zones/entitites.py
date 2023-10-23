import random
from typing import List, Tuple

from marl_factory_grid.environment.entity.object import _Object


class Zone(_Object):

    @property
    def positions(self):
        return self.coords

    def __init__(self, coords: List[Tuple[(int, int)]], *args, **kwargs):
        super(Zone, self).__init__(*args, **kwargs)
        self.coords = coords

    @property
    def random_pos(self):
        return random.choice(self.coords)
