from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.entity.object import Object
from marl_factory_grid.modules.batteries import constants as b
from marl_factory_grid.utils.utility_classes import RenderEntity


class Battery(Object):

    @property
    def var_can_be_bound(self):
        return True

    @property
    def is_discharged(self):
        return self.charge_level == 0

    @property
    def obs_tag(self):
        return self.name

    @property
    def encoding(self):
        return self.charge_level

    def __init__(self, initial_charge_level: float, owner: Entity, *args, **kwargs):
        super(Battery, self).__init__(*args, **kwargs)
        self.charge_level = initial_charge_level
        self.bind_to(owner)

    def do_charge_action(self, amount):
        if self.charge_level < 1:
            # noinspection PyTypeChecker
            self.charge_level = min(1, amount + self.charge_level)
            return c.VALID
        else:
            return c.NOT_VALID

    def decharge(self, amount) -> float:
        if self.charge_level != 0:
            # noinspection PyTypeChecker
            self.charge_level = max(0, amount + self.charge_level)
            return c.VALID
        else:
            return c.NOT_VALID

    def summarize_state(self):
        summary = super().summarize_state()
        summary.update(dict(belongs_to=self._bound_entity.name, chargeLevel=self.charge_level))
        return summary


class ChargePod(Entity):

    @property
    def encoding(self):
        return b.CHARGE_POD_SYMBOL

    def __init__(self, *args, charge_rate: float = 0.4,
                 multi_charge: bool = False, **kwargs):
        super(ChargePod, self).__init__(*args, **kwargs)
        self.charge_rate = charge_rate
        self.multi_charge = multi_charge

    def charge_battery(self, battery: Battery):
        if battery.charge_level == 1.0:
            return c.NOT_VALID
        if sum(1 for key, val in self.state.entities.pos_dict[self.pos] for guest in val if
               'agent' in guest.name.lower()) > 1:
            return c.NOT_VALID
        valid = battery.do_charge_action(self.charge_rate)
        return valid

    def render(self):
        return RenderEntity(b.CHARGE_PODS, self.pos)

    def summarize_state(self) -> dict:
        summery = super().summarize_state()
        summery.update(charge_rate=self.charge_rate)
        return summery
