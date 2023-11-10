from typing import Union, List, Tuple, Dict

from marl_factory_grid.environment.groups.collection import Collection
from .entities import Maintainer
from ..machines import constants as mc
from ..machines.actions import MachineAction


class Maintainers(Collection):
    _entity = Maintainer

    var_can_collide = True
    var_can_move = True
    var_is_blocking_light = False
    var_has_position = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def spawn(self, coords_or_quantity: Union[int, List[Tuple[(int, int)]]], *entity_args):
        self.add_items([self._entity(mc.MACHINES, MachineAction(), pos) for pos in coords_or_quantity])
