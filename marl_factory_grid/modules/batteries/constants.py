# Battery Env
CHARGE_PODS          = 'ChargePods'
BATTERIES            = 'Batteries'
BATTERY_DISCHARGED   = 'DISCHARGED'
CHARGE_POD_SYMBOL    = 1

ACTION_CHARGE                    = 'do_charge_action'

REWARD_CHARGE_VALID: float       = 0.1
Reward_CHARGE_FAIL: float        = -0.1
REWARD_BATTERY_DISCHARGED: float = -1.0
REWARD_DISCHARGE_DONE: float     = -1.0


GROUPED = "single"
SINGLE  = "grouped"
MODES = [GROUPED, SINGLE]
