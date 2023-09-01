from typing import List, Union
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult

from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.batteries import constants as b, rewards as r


class Btry(Rule):

    def __init__(self, initial_charge: float = 0.8, per_action_costs: Union[dict, float] = 0.02):
        super().__init__()
        self.per_action_costs = per_action_costs
        self.initial_charge = initial_charge

    def on_init(self, state, lvl_map):
        state[b.BATTERIES].spawn(state[c.AGENT], self.initial_charge)

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


class PodRules(Rule):

    def __init__(self, n_pods: int, charge_rate: float = 0.4, multi_charge: bool = False):
        super().__init__()
        self.multi_charge = multi_charge
        self.charge_rate = charge_rate
        self.n_pods = n_pods

    def on_init(self, state, lvl_map):
        pod_collection = state[b.CHARGE_PODS]
        empty_tiles = state[c.FLOOR].empty_tiles[:self.n_pods]
        pods = pod_collection.from_tiles(empty_tiles, entity_kwargs=dict(
            multi_charge=self.multi_charge, charge_rate=self.charge_rate)
                                         )
        pod_collection.add_items(pods)
