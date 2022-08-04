from typing import NamedTuple, Union

from environments.helpers import Constants as BaseConstants, EnvActions as BaseActions


class Constants(BaseConstants):
    # Battery Env
    CHARGE_PODS          = 'Charge_Pod'
    BATTERIES            = 'BATTERIES'
    BATTERY_DISCHARGED   = 'DISCHARGED'
    CHARGE_POD           = 1


class Actions(BaseActions):
    CHARGE              = 'do_charge_action'


class RewardsBtry(NamedTuple):
    CHARGE_VALID: float        = 0.1
    CHARGE_FAIL: float         = -0.1
    BATTERY_DISCHARGED: float  = -1.0


class BatteryProperties(NamedTuple):
    initial_charge: float = 0.8             #
    charge_rate: float = 0.4                #
    charge_locations: int = 20               #
    per_action_costs: Union[dict, float] = 0.02
    done_when_discharged: bool = False
    multi_charge: bool = False
