from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.entity.object import Object
from marl_factory_grid.modules.batteries import constants as b
from marl_factory_grid.utils.utility_classes import RenderEntity


class Battery(Object):

    @property
    def var_can_be_bound(self):
        return True

    @property
    def is_discharged(self) -> bool:
        """
        Indicates whether the Batteries charge level is at 0 or not.

        :return: Whether this battery is empty.
        """
        return self.charge_level == 0

    @property
    def obs_tag(self):
        return self.name

    @property
    def encoding(self):
        return self.charge_level

    def __init__(self, initial_charge_level, owner, *args, **kwargs):
        """
        Represents a battery entity in the environment that can be bound to an agent and charged at chargepods.

        :param initial_charge_level: The current charge level of the battery, ranging from 0 to 1.
        :type initial_charge_level: float

        :param owner: The entity to which the battery is bound.
        :type owner: Entity
        """
        super(Battery, self).__init__(*args, **kwargs)
        self.charge_level = initial_charge_level
        self.bind_to(owner)

    def do_charge_action(self, amount) -> bool:
        """
        Updates the Battery's charge level accordingly.

        :param amount: Amount added to the Battery's charge level.
        :returns: whether the battery could be charged. if not, it was already fully charged.
        """
        if self.charge_level < 1:
            # noinspection PyTypeChecker
            self.charge_level = min(1, amount + self.charge_level)
            return c.VALID
        else:
            return c.NOT_VALID

    def decharge(self, amount) -> bool:
        """
        Decreases the charge value of a battery. Currently only riggered by the battery-decharge rule.
        """
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

    def __init__(self, *args, charge_rate: float = 0.4, multi_charge: bool = False, **kwargs):
        """
        Represents a charging pod for batteries in the environment.

        :param charge_rate: The rate at which the charging pod charges batteries. Default is 0.4.
        :type charge_rate: float

        :param multi_charge: Indicates whether the charging pod supports charging multiple batteries simultaneously.
                        Default is False.
        :type multi_charge: bool
        """
        super(ChargePod, self).__init__(*args, **kwargs)
        self.charge_rate = charge_rate
        self.multi_charge = multi_charge

    def charge_battery(self, entity, state) -> bool:
        """
        Checks whether the battery can be charged. If so, triggers the charge action.

        :returns: whether the action was successful (valid) or not.
        """
        battery = state[b.BATTERIES].by_entity(entity)
        if battery.charge_level >= 1.0:
            return c.NOT_VALID
        if len([x for x in state[c.AGENT].by_pos(entity.pos)]) > 1:
            return c.NOT_VALID
        valid = battery.do_charge_action(self.charge_rate)
        return valid

    def render(self):
        return RenderEntity(b.CHARGE_PODS, self.pos)

    def summarize_state(self) -> dict:
        summary = super().summarize_state()
        summary.update(charge_rate=self.charge_rate)
        return summary
