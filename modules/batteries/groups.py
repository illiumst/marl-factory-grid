from environment.groups.env_objects import EnvObjects
from environment.groups.mixins import PositionMixin, HasBoundedMixin
from modules.batteries.entitites import ChargePod, Battery


class Batteries(HasBoundedMixin, EnvObjects):

    _entity = Battery
    is_blocking_light: bool = False
    can_collide: bool = False

    @property
    def obs_tag(self):
        return self.__class__.__name__

    @property
    def obs_pairs(self):
        return [(x.name, x) for x in self]

    def __init__(self, *args, **kwargs):
        super(Batteries, self).__init__(*args, **kwargs)

    def spawn_batteries(self, agents, initial_charge_level):
        batteries = [self._entity(initial_charge_level, agent) for _, agent in enumerate(agents)]
        self.add_items(batteries)


class ChargePods(PositionMixin, EnvObjects):

    _entity = ChargePod

    def __init__(self, *args, **kwargs):
        super(ChargePods, self).__init__(*args, **kwargs)

    def __repr__(self):
        return super(ChargePods, self).__repr__()
