from mfg_package.environment.groups.env_objects import EnvObjects
from mfg_package.environment.groups.mixins import PositionMixin
from mfg_package.modules.machines.entitites import Machine


class Machines(PositionMixin, EnvObjects):

    _entity = Machine
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super(Machines, self).__init__(*args, **kwargs)
