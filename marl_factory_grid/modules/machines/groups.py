from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.environment.groups.mixins import PositionMixin

from .entitites import Machine


class Machines(PositionMixin, Collection):

    _entity = Machine
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super(Machines, self).__init__(*args, **kwargs)
