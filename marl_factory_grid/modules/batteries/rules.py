from typing import List, Union

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.modules.batteries import constants as b
from marl_factory_grid.utils.results import TickResult, DoneResult


class BatteryDecharge(Rule):

    def __init__(self, initial_charge: float = 0.8, per_action_costs: Union[dict, float] = 0.02,
                 battery_charge_reward: float = b.REWARD_CHARGE_VALID,
                 battery_failed_reward: float = b.Reward_CHARGE_FAIL,
                 battery_discharge_reward: float = b.REWARD_BATTERY_DISCHARGED,
                 paralyze_agents_on_discharge: bool = False):
        f"""
        Enables the Battery Charge/Discharge functionality.

        :type paralyze_agents_on_discharge: bool
        :param paralyze_agents_on_discharge: Wether agents are still able to perform actions when discharged.
        :type per_action_costs: Union[dict, float] = 0.02
        :param per_action_costs: 1. dict: with an action name as key, provide a value for each 
                                    (maybe walking is less tedious as opening a door? Just saying...).
                                 2. float: each action "costs" the same.
        ----                         
         !!! Does not introduce any Env.-Done condition. 
         !!! Batteries can only be charged if agent posses the "Charge" Action.                 
         !!! Batteries can only be charged if there are "Charge Pods" and they are spawned!                      
        ----                         
        :type initial_charge: float
        :param initial_charge: How much juice they have.
        :type battery_discharge_reward: float
        :param battery_discharge_reward: Negative reward, when agents let their batters discharge. 
                                         Default: {b.REWARD_BATTERY_DISCHARGED}
        :type battery_failed_reward: float
        :param battery_failed_reward: Negative reward, when agent cannot charge, but do (overcharge, not on station).
                                       Default: {b.Reward_CHARGE_FAIL}
        :type battery_charge_reward: float
        :param battery_charge_reward: Positive reward, when agent actually charge their battery.
                                       Default: {b.REWARD_CHARGE_VALID}
        """
        super().__init__()
        self.paralyze_agents_on_discharge = paralyze_agents_on_discharge
        self.battery_discharge_reward = battery_discharge_reward
        self.battery_failed_reward = battery_failed_reward
        self.battery_charge_reward = battery_charge_reward
        self.per_action_costs = per_action_costs
        self.initial_charge = initial_charge

    def tick_step(self, state) -> List[TickResult]:
        batteries = state[b.BATTERIES]
        results = []

        for agent in state[c.AGENT]:
            if isinstance(self.per_action_costs, dict):
                energy_consumption = self.per_action_costs[agent.step_result()['action']]
            else:
                energy_consumption = self.per_action_costs

            batteries.by_entity(agent).decharge(energy_consumption)

            results.append(TickResult(self.name, entity=agent, validity=c.VALID, value=energy_consumption))

        return results

    def tick_post_step(self, state) -> List[TickResult]:
        results = []
        for btry in state[b.BATTERIES]:
            if btry.is_discharged:
                state.print(f'Battery of {btry.bound_entity.name} is discharged!')
                results.append(
                    TickResult(self.name, entity=btry.bound_entity, reward=self.battery_discharge_reward,
                               validity=c.VALID)
                )
                if self.paralyze_agents_on_discharge:
                    btry.bound_entity.paralyze(self.name)
                    results.append(
                        TickResult("Paralyzed", entity=btry.bound_entity, validity=c.VALID)
                    )
                    state.print(f'{btry.bound_entity.name} has just been paralyzed!')
            if btry.bound_entity.var_is_paralyzed and not btry.is_discharged:
                btry.bound_entity.de_paralyze(self.name)
                results.append(
                    TickResult("De-Paralyzed", entity=btry.bound_entity, validity=c.VALID)
                )
                state.print(f'{btry.bound_entity.name} has just been de-paralyzed!')
        return results


class DoneAtBatteryDischarge(BatteryDecharge):

    def __init__(self, reward_discharge_done=b.REWARD_DISCHARGE_DONE, mode: str = b.SINGLE, **kwargs):
        f"""
        Enables the Battery Charge/Discharge functionality. Additionally 

        :type mode: str
        :param mode: Does this Done rule trigger, when any battery is or all batteries are discharged? 
        :type per_action_costs: Union[dict, float] = 0.02
        :param per_action_costs: 1. dict: with an action name as key, provide a value for each 
                                    (maybe walking is less tedious as opening a door? Just saying...).
                                 2. float: each action "costs" the same.
                                 
        :type initial_charge: float
        :param initial_charge: How much juice they have.
        :type reward_discharge_done: float
        :param reward_discharge_done: Global negative reward, when agents let their batters discharge. 
                                         Default: {b.REWARD_BATTERY_DISCHARGED}
        :type battery_discharge_reward: float
        :param battery_discharge_reward: Negative reward, when agents let their batters discharge. 
                                         Default: {b.REWARD_BATTERY_DISCHARGED}
        :type battery_failed_reward: float
        :param battery_failed_reward: Negative reward, when agent cannot charge, but do (overcharge, not on station).
                                       Default: {b.Reward_CHARGE_FAIL}
        :type battery_charge_reward: float
        :param battery_charge_reward: Positive reward, when agent actually charge their battery.
                                       Default: {b.REWARD_CHARGE_VALID}
        """
        super().__init__(**kwargs)
        self.mode = mode
        self.reward_discharge_done = reward_discharge_done

    def on_check_done(self, state) -> List[DoneResult]:
        any_discharged = (self.mode == b.SINGLE and any(battery.is_discharged for battery in state[b.BATTERIES]))
        all_discharged = (self.mode == b.SINGLE and all(battery.is_discharged for battery in state[b.BATTERIES]))
        if any_discharged or all_discharged:
            return [DoneResult(self.name, validity=c.VALID, reward=self.reward_discharge_done)]
        else:
            return [DoneResult(self.name, validity=c.NOT_VALID)]
