from typing import List

from .entities import Maintainer
from marl_factory_grid.environment.entity.wall_floor import Floor
from marl_factory_grid.environment.groups.env_objects import Collection
from marl_factory_grid.environment.groups.mixins import PositionMixin
from ..machines.actions import MachineAction
from ...utils.states import Gamestate

from ..machines import constants as mc


class Maintainers(PositionMixin, Collection):

    _entity = Maintainer
    var_can_collide = True
    var_can_move = True
    var_is_blocking_light = False
    var_has_position = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def spawn(self, position, state: Gamestate):
        self.add_items([self._entity(state, mc.MACHINES, MachineAction(), pos) for pos in position])
