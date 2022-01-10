from typing import Union, NamedTuple, Dict, List

import numpy as np

from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Entity, EnvObject, BoundingMixin
from environments.factory.base.registers import EntityRegister, EnvObjectRegister
from environments.factory.base.renderer import RenderEntity
from environments.helpers import Constants as BaseConstants
from environments.helpers import EnvActions as BaseActions
from environments.helpers import Rewards as BaseRewards

from environments import helpers as h


class Constants(BaseConstants):
    # Battery Env
    CHARGE_PODS          = 'Charge_Pod'
    BATTERIES            = 'BATTERIES'
    BATTERY_DISCHARGED   = 'DISCHARGED'
    CHARGE_POD           = 1


class Actions(BaseActions):
    CHARGE              = 'do_charge_action'


class Rewards(BaseRewards):
    CHARGE_VALID        = 0.1
    CHARGE_FAIL         = -0.1
    BATTERY_DISCHARGED  = -1.0


class BatteryProperties(NamedTuple):
    initial_charge: float = 0.8             #
    charge_rate: float = 0.4                #
    charge_locations: int = 20               #
    per_action_costs: Union[dict, float] = 0.02
    done_when_discharged = False
    multi_charge: bool = False


c = Constants
a = Actions
r = Rewards


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
            return dict(valid=c.VALID, action=a.CHARGE, reward=r.CHARGE_VALID)
        else:
            return dict(valid=c.NOT_VALID, action=a.CHARGE, reward=r.CHARGE_FAIL)

    def decharge(self, amount) -> c:
        if self.charge_level != 0:
            # noinspection PyTypeChecker
            self.charge_level = max(0, amount + self.charge_level)
            self._register.notify_change_to_value(self)
            return c.VALID
        else:
            return c.NOT_VALID

    def summarize_state(self, **_):
        attr_dict = {key: str(val) for key, val in self.__dict__.items() if not key.startswith('_') and key != 'data'}
        attr_dict.update(dict(name=self.name))
        return attr_dict


class BatteriesRegister(EnvObjectRegister):

    _accepted_objects = Battery

    def __init__(self, *args, **kwargs):
        super(BatteriesRegister, self).__init__(*args, individual_slices=True,
                                                is_blocking_light=False, can_be_shadowed=False, **kwargs)
        self.is_observable = True

    def spawn_batteries(self, agents, initial_charge_level):
        batteries = [self._accepted_objects(initial_charge_level, agent, self) for _, agent in enumerate(agents)]
        self.register_additional_items(batteries)

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
        battery.do_charge_action(self.charge_rate)
        return c.VALID

    def summarize_state(self, n_steps=None) -> dict:
        if n_steps == h.STEPS_START:
            summary = super().summarize_state(n_steps=n_steps)
            return summary


class ChargePods(EntityRegister):

    _accepted_objects = ChargePod

    def __repr__(self):
        super(ChargePods, self).__repr__()


class BatteryFactory(BaseFactory):

    def __init__(self, *args, btry_prop=BatteryProperties(), **kwargs):
        if isinstance(btry_prop, dict):
            btry_prop = BatteryProperties(**btry_prop)
        self.btry_prop = btry_prop
        super().__init__(*args, **kwargs)

    def _additional_per_agent_raw_observations(self, agent) -> Dict[str, np.typing.ArrayLike]:
        additional_raw_observations = super()._additional_per_agent_raw_observations(agent)
        additional_raw_observations.update({c.BATTERIES: self[c.BATTERIES].as_array_by_entity(agent)})
        return additional_raw_observations

    def _additional_observations(self) -> Dict[str, np.typing.ArrayLike]:
        additional_observations = super()._additional_observations()
        additional_observations.update({c.CHARGE_PODS: self[c.CHARGE_PODS].as_array()})
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
        batteries.spawn_batteries(self[c.AGENT], self.btry_prop.initial_charge)
        super_entities.update({c.BATTERIES: batteries, c.CHARGE_PODS: charge_pods})
        return super_entities

    def do_additional_step(self) -> (List[dict], dict):
        super_reward_info = super(BatteryFactory, self).do_additional_step()

        # Decharge
        batteries = self[c.BATTERIES]

        for agent in self[c.AGENT]:
            if isinstance(self.btry_prop.per_action_costs, dict):
                energy_consumption = self.btry_prop.per_action_costs[agent.temp_action]
            else:
                energy_consumption = self.btry_prop.per_action_costs

            batteries.by_entity(agent).decharge(energy_consumption)

        return super_reward_info

    def do_charge_action(self, agent) -> (dict, dict):
        if charge_pod := self[c.CHARGE_PODS].by_pos(agent.pos):
            valid = charge_pod.charge_battery(self[c.BATTERIES].by_entity(agent))
            if valid:
                info_dict = {f'{agent.name}_{a.CHARGE}_VALID': 1}
                self.print(f'{agent.name} just charged batteries at {charge_pod.name}.')
            else:
                info_dict = {f'{agent.name}_{a.CHARGE}_FAIL': 1}
                self.print(f'{agent.name} failed to charged batteries at {charge_pod.name}.')
        else:
            valid = c.NOT_VALID
            info_dict = {f'{agent.name}_{a.CHARGE}_FAIL': 1}
            # info_dict = {f'{agent.name}_no_charger': 1}
            self.print(f'{agent.name} failed to charged batteries at {agent.pos}.')
        reward = dict(value=r.CHARGE_VALID if valid else r.CHARGE_FAIL, reason=a.CHARGE, info=info_dict)
        return valid, reward

    def do_additional_actions(self, agent: Agent, action: Action) -> (bool, dict):
        action_result = super().do_additional_actions(agent, action)
        if action_result is None:
            if action == a.CHARGE:
                action_result = self.do_charge_action(agent)
                return action_result
            else:
                return None
        else:
            return action_result
        pass

    def do_additional_reset(self) -> None:
        # There is Nothing to reset.
        pass

    def check_additional_done(self) -> (bool, dict):
        super_done, super_dict = super(BatteryFactory, self).check_additional_done()
        if super_done:
            return super_done, super_dict
        else:
            if self.btry_prop.done_when_discharged:
                if btry_done := any(battery.is_discharged for battery in self[c.BATTERIES]):
                    super_dict.update(DISCHARGE_DONE=1)
                    return btry_done, super_dict
                else:
                    pass
            else:
                pass
        pass

    def additional_per_agent_reward(self, agent: Agent) -> Dict[str, dict]:
        reward_event_dict = super(BatteryFactory, self).additional_per_agent_reward(agent)
        if self[c.BATTERIES].by_entity(agent).is_discharged:
            self.print(f'{agent.name} Battery is discharged!')
            info_dict = {f'{agent.name}_{c.BATTERY_DISCHARGED}': 1}
            reward_event_dict.update({c.BATTERY_DISCHARGED: {'reward': r.BATTERY_DISCHARGED, 'info': info_dict}})
        else:
            # All Fine
            pass
        return reward_event_dict

    def render_additional_assets(self):
        # noinspection PyUnresolvedReferences
        additional_assets = super().render_additional_assets()
        charge_pods = [RenderEntity(c.CHARGE_PODS, charge_pod.tile.pos) for charge_pod in self[c.CHARGE_PODS]]
        additional_assets.extend(charge_pods)
        return additional_assets
