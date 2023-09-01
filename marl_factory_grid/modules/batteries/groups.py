from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin, HasBoundMixin
from marl_factory_grid.modules.batteries.entitites import Pod, Battery


class Batteries(HasBoundMixin, EnvObjects):

    _entity = Battery
    is_blocking_light: bool = False
    can_collide: bool = False

    @property
    def obs_tag(self):
        return self.__class__.__name__

    def __init__(self, *args, **kwargs):
        super(Batteries, self).__init__(*args, **kwargs)

    def spawn(self, agents, initial_charge_level):
        batteries = [self._entity(initial_charge_level, agent) for _, agent in enumerate(agents)]
        self.add_items(batteries)


class ChargePods(PositionMixin, EnvObjects):

    _entity = Pod

    def __init__(self, *args, **kwargs):
        super(ChargePods, self).__init__(*args, **kwargs)

    def __repr__(self):
        return super(ChargePods, self).__repr__()
