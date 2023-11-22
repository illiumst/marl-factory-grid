from marl_factory_grid.environment.groups.collection import Collection

from .entitites import Machine


class Machines(Collection):

    _entity = Machine

    @property
    def var_can_collide(self):
        return False

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_has_position(self):
        return True

    def __init__(self, *args, **kwargs):
        """
        A Collection of Machines.
        """
        super(Machines, self).__init__(*args, **kwargs)

