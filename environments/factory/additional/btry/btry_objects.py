from environments import helpers as h
from environments.factory.base.objects import BoundingMixin, EnvObject, Entity
from environments.factory.additional.btry.btry_util import Constants as c


class Battery(BoundingMixin, EnvObject):

    @property
    def is_discharged(self):
        return self.charge_level == 0

    def __init__(self, initial_charge_level: float, *args, **kwargs):
        super(Battery, self).__init__(*args, **kwargs)
        self.charge_level = initial_charge_level

    def encoding(self):
        return self.charge_level

    def do_charge_action(self, amount):
        if self.charge_level < 1:
            # noinspection PyTypeChecker
            self.charge_level = min(1, amount + self.charge_level)
            return c.VALID
        else:
            return c.NOT_VALID

    def decharge(self, amount) -> c:
        if self.charge_level != 0:
            # noinspection PyTypeChecker
            self.charge_level = max(0, amount + self.charge_level)
            self._collection.notify_change_to_value(self)
            return c.VALID
        else:
            return c.NOT_VALID

    def summarize_state(self, **_):
        attr_dict = {key: str(val) for key, val in self.__dict__.items() if not key.startswith('_') and key != 'data'}
        attr_dict.update(dict(name=self.name))
        return attr_dict


class ChargePod(Entity):

    @property
    def encoding(self):
        return c.CHARGE_POD

    def __init__(self, *args, charge_rate: float = 0.4,
                 multi_charge: bool = False, **kwargs):
        super(ChargePod, self).__init__(*args, **kwargs)
        self.charge_rate = charge_rate
        self.multi_charge = multi_charge

    def charge_battery(self, battery: Battery):
        if battery.charge_level == 1.0:
            return c.NOT_VALID
        if sum(guest for guest in self.tile.guests if 'agent' in guest.name.lower()) > 1:
            return c.NOT_VALID
        valid = battery.do_charge_action(self.charge_rate)
        return valid

    def summarize_state(self, n_steps=None) -> dict:
        if n_steps == h.STEPS_START:
            summary = super().summarize_state(n_steps=n_steps)
            return summary
        else:
            {}
