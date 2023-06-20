from typing import NamedTuple, Union

# Battery Env
CHARGE_PODS          = 'ChargePods'
BATTERIES            = 'Batteries'
BATTERY_DISCHARGED   = 'DISCHARGED'
CHARGE_POD_SYMBOL    = 1


CHARGE              = 'do_charge_action'


class BatteryProperties(NamedTuple):
    initial_charge: float = 0.8             #
    charge_rate: float = 0.4                #
    charge_locations: int = 20               #
    per_action_costs: Union[dict, float] = 0.02
    done_when_discharged: bool = False
    multi_charge: bool = False
