from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.batteries.entitites import Pod, Battery


class Batteries(Collection):

    _entity = Battery

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_move(self):
        return False

    @property
    def var_has_position(self):
        return True

    @property
    def obs_tag(self):
        return self.__class__.__name__

    def __init__(self, *args, **kwargs):
        super(Batteries, self).__init__(*args, **kwargs)

    def spawn(self, agents, initial_charge_level):
        batteries = [self._entity(initial_charge_level, agent) for _, agent in enumerate(agents)]
        self.add_items(batteries)


class ChargePods(Collection):

    _entity = Pod

    def __init__(self, *args, **kwargs):
        super(ChargePods, self).__init__(*args, **kwargs)

    def __repr__(self):
        return super(ChargePods, self).__repr__()
