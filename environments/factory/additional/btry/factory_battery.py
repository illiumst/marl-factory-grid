from typing import Dict, List

import numpy as np

from environments.factory.additional.btry.btry_collections import BatteriesRegister, ChargePods
from environments.factory.additional.btry.btry_util import Constants, Actions, RewardsBtry, BatteryProperties
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action
from environments.factory.base.renderer import RenderEntity

c = Constants
a = Actions


class BatteryFactory(BaseFactory):

    def __init__(self, *args, btry_prop=BatteryProperties(), rewards_btry: RewardsBtry = RewardsBtry(),
                 **kwargs):
        if isinstance(btry_prop, dict):
            btry_prop = BatteryProperties(**btry_prop)
        if isinstance(rewards_btry, dict):
            rewards_btry = RewardsBtry(**rewards_btry)
        self.btry_prop = btry_prop
        self.rewards_dest = rewards_btry
        super().__init__(*args, **kwargs)

    def per_agent_raw_observations_hook(self, agent) -> Dict[str, np.typing.ArrayLike]:
        additional_raw_observations = super().per_agent_raw_observations_hook(agent)
        additional_raw_observations.update({c.BATTERIES: self[c.BATTERIES].as_array_by_entity(agent)})
        return additional_raw_observations

    def observations_hook(self) -> Dict[str, np.typing.ArrayLike]:
        additional_observations = super().observations_hook()
        additional_observations.update({c.CHARGE_PODS: self[c.CHARGE_PODS].as_array()})
        return additional_observations

    @property
    def entities_hook(self):
        super_entities = super().entities_hook

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

    def step_hook(self) -> (List[dict], dict):
        super_reward_info = super(BatteryFactory, self).step_hook()

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
        reward = dict(value=self.rewards_dest.CHARGE_VALID if valid else self.rewards_dest.CHARGE_FAIL,
                      reason=a.CHARGE, info=info_dict)
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

    def reset_hook(self) -> (List[dict], dict):
        super_reward_info = super(BatteryFactory, self).reset_hook()
        # There is Nothing to reset.
        return super_reward_info

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
        return super_done, super_dict

    def per_agent_reward_hook(self, agent: Agent) -> List[dict]:
        reward_event_list = super(BatteryFactory, self).per_agent_reward_hook(agent)
        if self[c.BATTERIES].by_entity(agent).is_discharged:
            self.print(f'{agent.name} Battery is discharged!')
            info_dict = {f'{agent.name}_{c.BATTERY_DISCHARGED}': 1}
            reward_event_list.append({'value': self.rewards_dest.BATTERY_DISCHARGED,
                                      'reason': c.BATTERY_DISCHARGED,
                                      'info': info_dict}
                                     )
        else:
            # All Fine
            pass
        return reward_event_list

    def render_assets_hook(self):
        # noinspection PyUnresolvedReferences
        additional_assets = super().render_assets_hook()
        charge_pods = [RenderEntity(c.CHARGE_PODS, charge_pod.tile.pos) for charge_pod in self[c.CHARGE_PODS]]
        additional_assets.extend(charge_pods)
        return additional_assets
