from typing import List, Union
from environment.rules import Rule
from environment.utils.results import TickResult, DoneResult

from environment import constants as c
from modules.batteries import constants as b, rewards as r


class Btry(Rule):

    def __init__(self, initial_charge: float = 0.8, per_action_costs: Union[dict, float] = 0.02):
        super().__init__()
        self.per_action_costs = per_action_costs
        self.initial_charge = initial_charge

    def on_init(self, state):
        state[b.BATTERIES].spawn_batteries(state[c.AGENT], self.initial_charge)

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        # Decharge
        batteries = state[b.BATTERIES]
        results = []

        for agent in state[c.AGENT]:
            if isinstance(self.per_action_costs, dict):
                energy_consumption = self.per_action_costs[agent.step_result()['action']]
            else:
                energy_consumption = self.per_action_costs

            batteries.by_entity(agent).decharge(energy_consumption)

            results.append(TickResult(self.name, reward=0, entity=agent, validity=c.VALID))

        return results

    def tick_post_step(self, state) -> List[TickResult]:
        results = []
        for btry in state[b.BATTERIES]:
            if btry.is_discharged:
                state.print(f'Battery of {btry.bound_entity.name} is discharged!')
                results.append(
                    TickResult(self.name, entity=btry.bound_entity, reward=r.BATTERY_DISCHARGED, validity=c.VALID))
        else:
            pass
        return results


class BtryDoneAtDischarge(Rule):

    def __init__(self):
        super().__init__()

    def on_check_done(self, state) -> List[DoneResult]:
        if btry_done := any(battery.is_discharged for battery in state[b.BATTERIES]):
            return [DoneResult(self.name, validity=c.VALID, reward=r.BATTERY_DISCHARGED)]
        else:
            return [DoneResult(self.name, validity=c.NOT_VALID, reward=0)]

