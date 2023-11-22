from typing import Union, List, Tuple, Dict

from marl_factory_grid.environment.groups.collection import Collection
from .entities import Maintainer
from ..machines import constants as mc
from ..machines.actions import MachineAction


class Maintainers(Collection):
    _entity = Maintainer

    @property
    def var_can_collide(self):
        return True

    @property
    def var_can_move(self):
        return True

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_has_position(self):
        return True

    def __init__(self, *args, **kwargs):
        """
        A collection of maintainers
        """
        super().__init__(*args, **kwargs)

    def spawn(self, coords_or_quantity: Union[int, List[Tuple[(int, int)]]], *entity_args):
        self.add_items([self._entity(mc.MACHINES, MachineAction(), pos) for pos in coords_or_quantity])
