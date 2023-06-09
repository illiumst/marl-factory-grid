from environment.groups.env_objects import EnvObjects
from environment.groups.mixins import PositionMixin
from modules.machines.entitites import Machine


class Machines(PositionMixin, EnvObjects):

    _entity = Machine
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super(Machines, self).__init__(*args, **kwargs)
