# Names
DANGER_ZONE             = 'x'                   # Dange Zone tile _identifier for resolving the string based map files.
DEFAULTS                = 'Defaults'
SELF                    = 'Self'
PLACEHOLDER             = 'Placeholder'
FLOOR                   = 'Floor'               # Identifier of Floor-objects and groups (groups).
FLOORS                  = 'Floors'              # Identifier of Floor-objects and groups (groups).
WALL                    = 'Wall'                # Identifier of Wall-objects and groups (groups).
WALLS                   = 'Walls'               # Identifier of Wall-objects and groups (groups).
LEVEL                   = 'Level'               # Identifier of Level-objects and groups (groups).
AGENT                   = 'Agent'               # Identifier of Agent-objects and groups (groups).
OTHERS                  = 'Other'
COMBINED                = 'Combined'
GLOBALPOSITIONS        = 'GlobalPositions'     # Identifier of the global position slice

# Attributes
IS_BLOCKING_LIGHT       = 'var_is_blocking_light'
HAS_POSITION            = 'var_has_position'
HAS_NO_POSITION         = 'has_no_position'
ALL                     = 'All'

# Symbols (Read from map-files)
SYMBOL_WALL             = '#'
SYMBOL_FLOOR            = '-'

VALID                   = True            # Identifier to rename boolean values in the context of actions.
NOT_VALID               = False           # Identifier to rename boolean values in the context of actions.
VALUE_FREE_CELL         = 0               # Free-Cell value used in observation
VALUE_OCCUPIED_CELL     = 1               # Occupied-Cell value used in observation
VALUE_NO_POS            = (-9999, -9999)  # Invalid Position value used in the environment (smth. is off-grid)


ACTION                  = 'action'  # Identifier of Action-objects and groups (groups).
COLLISION               = 'Collision'  # Identifier to use in the context of collitions.
LAST_POS                = 'LAST_POS'  # Identifiert for retrieving an enitites last pos.
VALIDITY                = 'VALIDITY'  # Identifiert for retrieving the Validity of Action, Tick, etc. ...

# Actions
# Movements
NORTH                   = 'north'
EAST                    = 'east'
SOUTH                   = 'south'
WEST                    = 'west'
NORTHEAST               = 'north_east'
SOUTHEAST               = 'south_east'
SOUTHWEST               = 'south_west'
NORTHWEST               = 'north_west'

# Move Groups
MOVE8                   = 'Move8'
MOVE4                   = 'Move4'

# No-Action / Wait
NOOP                    = 'Noop'

# Result Identifier
MOVEMENTS_VALID = 'motion_valid'
MOVEMENTS_FAIL  = 'motion_not_valid'
