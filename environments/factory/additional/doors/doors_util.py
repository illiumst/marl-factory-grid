
from typing import NamedTuple

from environments.helpers import Constants as BaseConstants, EnvActions as BaseActions


class Constants(BaseConstants):
    DOOR         = 'Door'   # Identifier of Single-Door Entities.
    DOORS        = 'Doors'  # Identifier of Door-objects and sets (collections).
    DOOR_SYMBOL  = 'D'                   # Door identifier for resolving the string based map files.

    ACCESS_DOOR_CELL = 1 / 3  # Access-door-Cell value used in observation
    OPEN_DOOR_CELL = 2 / 3  # Open-door-Cell value used in observation
    CLOSED_DOOR_CELL = 3 / 3  # Closed-door-Cell value used in observation

    CLOSED_DOOR         = 'closed'              # Identifier to compare door-is-closed state
    OPEN_DOOR           = 'open'                # Identifier to compare door-is-open state
    # ACCESS_DOOR         = 'access'            # Identifier to compare access positions


class Actions(BaseActions):
    USE_DOOR = 'use_door'


class RewardsDoor(NamedTuple):
    USE_DOOR_VALID: float  = -0.00
    USE_DOOR_FAIL: float   = -0.01


class DoorProperties(NamedTuple):
    indicate_door_area: bool = True            # Wether the door area should be indicated in the agents' observation.
