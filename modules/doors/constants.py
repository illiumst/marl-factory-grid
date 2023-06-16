# Names / Identifiers
DOOR                    = 'Door'   # Identifier of Single-Door Entities.
DOORS                   = 'Doors'  # Identifier of Door-objects and groups (groups).

# Symbols (in map)
SYMBOL_DOOR             = 'D'                   # Door _identifier for resolving the string based map files.

# Values
VALUE_ACCESS_INDICATOR  = 1 / 3  # Access-door-Cell value used in observation
VALUE_OPEN_DOOR         = 2 / 3  # Open-door-Cell value used in observation
VALUE_CLOSED_DOOR       = 3 / 3  # Closed-door-Cell value used in observation

# States
STATE_CLOSED            = 'closed'              # Identifier to compare door-is-closed state
STATE_OPEN              = 'open'                # Identifier to compare door-is-open state

# Actions
ACTION_DOOR_USE         = 'use_door'