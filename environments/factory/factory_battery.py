from typing import Union, NamedTuple, Dict

import numpy as np

from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Entity, EnvObject, BoundingMixin
from environments.factory.base.registers import EntityRegister, EnvObjectRegister
from environments.factory.base.renderer import RenderEntity
from environments.helpers import Constants as c, Constants

from environments import helpers as h


CHARGE_ACTION = h.EnvActions.CHARGE
CHARGE_POD = 1


class BatteryProperties(NamedTuple):
    initial_charge: float = 0.8             #
    charge_rate: float = 0.4                #
    charge_locations: int = 20               #
    per_action_costs: Union[dict, float] = 0.02
    done_when_discharged = False
    multi_charge: bool = False


class Battery(EnvObject, BoundingMixin):

    @property
    def is_discharged(self):
        return self.charge_level == 0

    def __init__(self, initial_charge_level: float, *args, **kwargs):
        super(Battery, self).__init__(*args, **kwargs)
        self.charge_level = initial_charge_level

    def encoding(self):
        return self.charge_level

    def charge(self, amount) -> c:
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
            self._register.notify_change_to_value(self)
            return c.VALID
        else:
            return c.NOT_VALID

    def summarize_state(self, **kwargs):
        attr_dict = {key: str(val) for key, val in self.__dict__.items() if not key.startswith('_') and key != 'data'}
        attr_dict.update(dict(name=self.name))
        return attr_dict


class BatteriesRegister(EnvObjectRegister):

    _accepted_objects = Battery
    is_blocking_light = False
    can_be_shadowed = False
    hide_from_obs_builder = True

    def __init__(self, *args, **kwargs):
        super(BatteriesRegister, self).__init__(*args, is_per_agent=True, individual_slices=True, **kwargs)
        self.is_observable = True

    def as_array(self):
        # ToDO: Make this Lazy
        self._array[:] = c.FREE_CELL.value
        for inv_idx, battery in enumerate(self):
            self._array[inv_idx] = battery.as_array()
        return self._array

    def spawn_batteries(self, agents, pomdp_r, initial_charge_level):
        batteries = [self._accepted_objects(pomdp_r, self._shape, agent,
                                            initial_charge_level)
                     for _, agent in enumerate(agents)]
        self.register_additional_items(batteries)

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, bat in enumerate(self) if bat.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def by_entity(self, entity):
        try:
            return next((bat for bat in self if bat.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def summarize_states(self, n_steps=None):
        # as dict with additional nesting
        # return dict(items=super(Inventories, self).summarize_states())
        return super(BatteriesRegister, self).summarize_states(n_steps=n_steps)


class ChargePod(Entity):

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        return CHARGE_POD

    def __init__(self, *args, charge_rate: float = 0.4,
                 multi_charge: bool = False, **kwargs):
        super(ChargePod, self).__init__(*args, **kwargs)
        self.charge_rate = charge_rate
        self.multi_charge = multi_charge

    def charge_battery(self, battery: Battery):
        if battery.charge_level == 1.0:
            return c.NOT_VALID
        if sum(guest for guest in self.tile.guests if c.AGENT.name in guest.name) > 1:
            return c.NOT_VALID
        battery.charge(self.charge_rate)
        return c.VALID

    def summarize_state(self, n_steps=None) -> dict:
        if n_steps == h.STEPS_START:
            summary = super().summarize_state(n_steps=n_steps)
            return summary


class ChargePods(EntityRegister):

    _accepted_objects = ChargePod

    @DeprecationWarning
    def Xas_array(self):
        self._array[:] = c.FREE_CELL.value
        for item in self:
            if item.pos != c.NO_POS.value:
                self._array[0, item.x, item.y] = item.encoding
        return self._array

    def __repr__(self):
        super(ChargePods, self).__repr__()


class BatteryFactory(BaseFactory):

    def __init__(self, *args, btry_prop=BatteryProperties(), **kwargs):
        if isinstance(btry_prop, dict):
            btry_prop = BatteryProperties(**btry_prop)
        self.btry_prop = btry_prop
        super().__init__(*args, **kwargs)

    def _additional_raw_observations(self, agent) -> Dict[Constants, np.typing.ArrayLike]:
        additional_raw_observations = super()._additional_raw_observations(agent)
        additional_raw_observations.update({c.BATTERIES: self[c.BATTERIES].by_entity(agent).as_array()})
        return additional_raw_observations

    def _additional_observations(self) -> Dict[Constants, np.typing.ArrayLike]:
        additional_observations = super()._additional_observations()
        additional_observations.update({c.CHARGE_POD: self[c.CHARGE_POD].as_array()})
        return additional_observations

    @property
    def additional_entities(self):
        super_entities = super().additional_entities

        empty_tiles = self[c.FLOOR].empty_tiles[:self.btry_prop.charge_locations]
        charge_pods = ChargePods.from_tiles(
            empty_tiles, self._level_shape,
            entity_kwargs=dict(charge_rate=self.btry_prop.charge_rate,
                               multi_charge=self.btry_prop.multi_charge)
        )

        batteries = BatteriesRegister(self._level_shape if not self._pomdp_r else ((self.pomdp_diameter,) * 2),
                                      )
        batteries.spawn_batteries(self[c.AGENT], self._pomdp_r, self.btry_prop.initial_charge)
        super_entities.update({c.BATTERIES: batteries, c.CHARGE_POD: charge_pods})
        return super_entities

    def do_additional_step(self) -> dict:
        info_dict = super(BatteryFactory, self).do_additional_step()

        # Decharge
        batteries = self[c.BATTERIES]

        for agent in self[c.AGENT]:
            if isinstance(self.btry_prop.per_action_costs, dict):
                energy_consumption = self.btry_prop.per_action_costs[agent.temp_action]
            else:
                energy_consumption = self.btry_prop.per_action_costs

            batteries.by_entity(agent).decharge(energy_consumption)

        return info_dict

    def do_charge(self, agent) -> c:
        if charge_pod := self[c.CHARGE_POD].by_pos(agent.pos):
            return charge_pod.charge_battery(self[c.BATTERIES].by_entity(agent))
        else:
            return c.NOT_VALID

    def do_additional_actions(self, agent: Agent, action: Action) -> Union[None, c]:
        valid = super().do_additional_actions(agent, action)
        if valid is None:
            if action == CHARGE_ACTION:
                valid = self.do_charge(agent)
                return valid
            else:
                return None
        else:
            return valid
        pass

    def do_additional_reset(self) -> None:
        # There is Nothing to reset.
        pass

    def check_additional_done(self) -> bool:
        super_done = super(BatteryFactory, self).check_additional_done()
        if super_done:
            return super_done
        else:
            return self.btry_prop.done_when_discharged and any(battery.is_discharged for battery in self[c.BATTERIES])
        pass

    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        reward, info_dict = super(BatteryFactory, self).calculate_additional_reward(agent)
        if h.EnvActions.CHARGE == agent.temp_action:
            if agent.temp_valid:
                charge_pod = self[c.CHARGE_POD].by_pos(agent.pos)
                info_dict.update({f'{agent.name}_charge': 1})
                info_dict.update(agent_charged=1)
                self.print(f'{agent.name} just charged batteries at {charge_pod.pos}.')
                reward += 0.1
            else:
                self[c.DROP_OFF].by_pos(agent.pos)
                info_dict.update({f'{agent.name}_failed_charge': 1})
                info_dict.update(failed_charge=1)
                self.print(f'{agent.name} just tried to charge at {agent.pos}, but failed.')
                reward -= 0.1

        if self[c.BATTERIES].by_entity(agent).is_discharged:
            info_dict.update({f'{agent.name}_discharged': 1})
            reward -= 1
        else:
            info_dict.update({f'{agent.name}_battery_level': self[c.BATTERIES].by_entity(agent).charge_level})
        return reward, info_dict

    def render_additional_assets(self):
        # noinspection PyUnresolvedReferences
        additional_assets = super().render_additional_assets()
        charge_pods = [RenderEntity(c.CHARGE_POD.value, charge_pod.tile.pos) for charge_pod in self[c.CHARGE_POD]]
        additional_assets.extend(charge_pods)
        return additional_assets

