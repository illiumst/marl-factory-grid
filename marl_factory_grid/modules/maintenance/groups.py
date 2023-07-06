from typing import List

from .entities import Maintainer
from marl_factory_grid.environment.entity.wall_floor import Floor
from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin
from ..machines.actions import MachineAction
from ...utils.render import RenderEntity
from ...utils.states import Gamestate

from ..machines import constants as mc
from . import constants as mi


class Maintainers(PositionMixin, EnvObjects):

    _entity = Maintainer
    var_can_collide = True
    var_can_move = True
    var_is_blocking_light = False
    var_has_position = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def spawn(self, tiles: List[Floor], state: Gamestate):
        self.add_items([self._entity(state, mc.MACHINES, MachineAction(), tile) for tile in tiles])
