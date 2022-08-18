from typing import NamedTuple

from environments.helpers import Constants as BaseConstants, EnvActions as BaseActions


class Constants(BaseConstants):
    DIRT = 'DirtPile'


class Actions(BaseActions):
    CLEAN_UP = 'do_cleanup_action'


class RewardsDirt(NamedTuple):
    CLEAN_UP_VALID: float          = 0.5
    CLEAN_UP_FAIL: float           = -0.1
    CLEAN_UP_LAST_PIECE: float     = 4.5


class DirtProperties(NamedTuple):
    initial_dirt_ratio: float = 0.3         # On INIT, on max how many tiles does the dirt spawn in percent.
    initial_dirt_spawn_r_var: float = 0.05  # How much does the dirt spawn amount vary?
    clean_amount: float = 1                 # How much does the robot clean with one actions.
    max_spawn_ratio: float = 0.20           # On max how many tiles does the dirt spawn in percent.
    max_spawn_amount: float = 0.3           # How much dirt does spawn per tile at max.
    spawn_frequency: int = 0                # Spawn Frequency in Steps.
    max_local_amount: int = 2               # Max dirt amount per tile.
    max_global_amount: int = 20             # Max dirt amount in the whole environment.
    dirt_smear_amount: float = 0.2          # Agents smear dirt, when not cleaning up in place.
    done_when_clean: bool = True
