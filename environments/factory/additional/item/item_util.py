from typing import NamedTuple

from environments.helpers import Constants as BaseConstants, EnvActions as BaseActions


class Constants(BaseConstants):
    NO_ITEM = 0
    ITEM_DROP_OFF = 1
    # Item Env
    ITEM                = 'Item'
    INVENTORY           = 'Inventory'
    DROP_OFF            = 'Drop_Off'


class Actions(BaseActions):
    ITEM_ACTION     = 'ITEMACTION'


class RewardsItem(NamedTuple):
    DROP_OFF_VALID: float = 0.1
    DROP_OFF_FAIL: float = -0.1
    PICK_UP_FAIL: float  = -0.1
    PICK_UP_VALID: float  = 0.1


class ItemProperties(NamedTuple):
    n_items:                         int  = 5     # How many items are there at the same time
    spawn_frequency:                 int  = 10     # Spawn Frequency in Steps
    n_drop_off_locations:            int  = 5     # How many DropOff locations are there at the same time
    max_dropoff_storage_size:        int  = 0     # How many items are needed until the dropoff is full
    max_agent_inventory_capacity:    int  = 5     # How many items are needed until the agent inventory is full
