from typing import NamedTuple

from environments.helpers import Constants as BaseConstants, EnvActions as BaseActions


class Constants(BaseConstants):
    # Destination Env
    DEST                    = 'Destination'
    DESTINATION             = 1
    DESTINATION_DONE        = 0.5
    DEST_REACHED            = 'ReachedDestination'


class Actions(BaseActions):
    WAIT_ON_DEST    = 'WAIT'


class RewardsDest(NamedTuple):

    WAIT_VALID: float      = 0.1
    WAIT_FAIL: float       = -0.1
    DEST_REACHED: float    = 5.0


class DestModeOptions(object):
    DONE        = 'DONE'
    GROUPED     = 'GROUPED'
    PER_DEST    = 'PER_DEST'


class DestProperties(NamedTuple):
    n_dests:                                     int = 1     # How many destinations are there
    dwell_time:                                  int = 0     # How long does the agent need to "wait" on a destination
    spawn_frequency:                             int = 0
    spawn_in_other_zone:                        bool = True  #
    spawn_mode:                                  str = DestModeOptions.DONE

    assert dwell_time >= 0, 'dwell_time cannot be < 0!'
    assert spawn_frequency >= 0, 'spawn_frequency cannot be < 0!'
    assert n_dests >= 0, 'n_destinations cannot be < 0!'
    assert (spawn_mode == DestModeOptions.DONE) != bool(spawn_frequency)
