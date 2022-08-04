from environments.factory.additional.btry.btry_objects import Battery, ChargePod
from environments.factory.base.registers import EnvObjectCollection, EntityCollection


class BatteriesRegister(EnvObjectCollection):

    _accepted_objects = Battery

    def __init__(self, *args, **kwargs):
        super(BatteriesRegister, self).__init__(*args, individual_slices=True,
                                                is_blocking_light=False, can_be_shadowed=False, **kwargs)
        self.is_observable = True

    def spawn_batteries(self, agents, initial_charge_level):
        batteries = [self._accepted_objects(initial_charge_level, agent, self) for _, agent in enumerate(agents)]
        self.add_additional_items(batteries)

    def summarize_states(self, n_steps=None):
        # as dict with additional nesting
        # return dict(items=super(Inventories, cls).summarize_states())
        return super(BatteriesRegister, self).summarize_states(n_steps=n_steps)

    # Todo Move this to Mixin!
    def by_entity(self, entity):
        try:
            return next((x for x in self if x.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, x in enumerate(self) if x.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def as_array_by_entity(self, entity):
        return self._array[self.idx_by_entity(entity)]


class ChargePods(EntityCollection):

    _accepted_objects = ChargePod

    def __repr__(self):
        super(ChargePods, self).__repr__()

    def summarize_states(self, n_steps=None):
        # as dict with additional nesting
        # return dict(items=super(Inventories, cls).summarize_states())
        return super(ChargePods, self).summarize_states(n_steps=n_steps)
