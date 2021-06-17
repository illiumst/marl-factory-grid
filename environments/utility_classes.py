from typing import Union, List, NamedTuple, Tuple
import numpy as np

from environments import helpers as h


IS_CLOSED = 'CLOSED'
IS_OPEN = 'OPEN'


class MovementProperties(NamedTuple):
    allow_square_movement: bool = True
    allow_diagonal_movement: bool = False
    allow_no_op: bool = False


# Preperations for Entities (not used yet)
class Entity:

    @property
    def pos(self):
        return self._pos

    @property
    def identifier(self):
        return self._identifier

    def __init__(self, identifier, pos):
        self._pos = pos
        self._identifier = identifier


class Door(Entity):

    @property
    def is_closed(self):
        return self._state == IS_CLOSED

    @property
    def is_open(self):
        return self._state == IS_OPEN

    @property
    def status(self):
        return self._state

    def __init__(self, *args, closed_on_init=True, **kwargs):
        super(Door, self).__init__(*args, **kwargs)
        self._state = IS_CLOSED if closed_on_init else IS_OPEN

    def use(self):
        self._state: str = IS_CLOSED if self._state == IS_OPEN else IS_OPEN
    pass


class Agent(Entity):

    @property
    def direction_of_vision(self):
        return self._direction_of_vision

    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        self._direction_of_vision = (None, None)

    def move(self, new_pos: Tuple[int, int]):
        x_old, y_old = self.pos
        self._pos = new_pos
        x_new, y_new = new_pos
        self._direction_of_vision = (x_old-x_new, y_old-y_new)
        return self.pos


class AgentState:

    @property
    def collisions(self):
        return np.argwhere(self.collision_vector != 0).flatten()

    @property
    def direction_of_view(self):
        last_x, last_y = self._last_pos
        curr_x, curr_y = self.pos
        return last_x-curr_x, last_y-curr_y

    def __init__(self, i: int, action: int):
        self.i = i
        self.action = action

        self.collision_vector = None
        self.action_valid = None
        self.pos = None
        self._last_pos = (-1, -1)

    def update(self, **kwargs):                             # is this hacky?? o.0
        last_pos = self.pos
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                raise AttributeError(f'"{key}" cannot be updated, this attr is not a part of {self.__name__}')
        if self.action_valid and last_pos != self.pos:
            self._last_pos = last_pos

    def reset(self):
        self.__init__(self.i, self.action)


class DoorState:

    def __init__(self, i: int, pos: Tuple[int, int], closed_on_init=True, auto_close_interval=10):
        self.i = i
        self.pos = pos
        self._state = self._state = IS_CLOSED if closed_on_init else IS_OPEN
        self.auto_close_interval = auto_close_interval
        self.time_to_close = -1

    @property
    def is_closed(self):
        return self._state == IS_CLOSED

    @property
    def is_open(self):
        return self._state == IS_OPEN

    @property
    def status(self):
        return self._state

    def use(self):
        if self._state == IS_OPEN:
            self._state = IS_CLOSED
        else:
            self._state = IS_OPEN
            self.time_to_close = self.auto_close_interval

class Register:

    @property
    def n(self):
        return len(self)

    def __init__(self):
        self._register = dict()

    def __len__(self):
        return len(self._register)

    def __add__(self, other: str):
        assert isinstance(other, str), f'All item names have to be of type {str}'
        self._register.update({len(self._register): other})
        return self

    def register_additional_items(self, others: List[str]):
        for other in others:
            self + other
        return self

    def keys(self):
        return self._register.keys()

    def values(self):
        return self._register.values()

    def items(self):
        return self._register.items()

    def __getitem__(self, item):
        try:
            return self._register[item]
        except KeyError:
            print('NO')
            raise

    def by_name(self, item):
        return list(self._register.keys())[list(self._register.values()).index(item)]

    def __repr__(self):
        return f'{self.__class__.__name__}({self._register})'


class Agents(Register):

    def __init__(self, n_agents):
        super(Agents, self).__init__()
        self.register_additional_items([f'agent#{i}' for i in range(n_agents)])
        self._agents = [Agent(x, (-1, -1)) for x in self.keys()]
        pass

    def __getitem__(self, item):
        return self._agents[item]

    def get_name(self, item):
        return self._register[item]

    def by_name(self, item):
        return self[super(Agents, self).by_name(item)]

    def __add__(self, other):
        super(Agents, self).__add__(other)
        self._agents.append(Agent(len(self)+1, (-1, -1)))


class Actions(Register):

    @property
    def movement_actions(self):
        return self._movement_actions

    def __init__(self, movement_properties: MovementProperties, can_use_doors=False):
        self.allow_no_op = movement_properties.allow_no_op
        self.allow_diagonal_movement = movement_properties.allow_diagonal_movement
        self.allow_square_movement = movement_properties.allow_square_movement
        self.can_use_doors = can_use_doors
        super(Actions, self).__init__()

        if self.allow_square_movement:
            self.register_additional_items(['north', 'east', 'south', 'west'])
        if self.allow_diagonal_movement:
            self.register_additional_items(['north_east', 'south_east', 'south_west', 'north_west'])
        self._movement_actions = self._register.copy()
        if self.can_use_doors:
            self.register_additional_items(['use_door'])
        if self.allow_no_op:
            self.register_additional_items(['no-op'])

    def is_moving_action(self, action: Union[str, int]):
        if isinstance(action, str):
            return action in self.movement_actions.values()
        else:
            return self[action] in self.movement_actions.values()

    def is_no_op(self, action: Union[str, int]):
        if isinstance(action, str):
            action = self.by_name(action)
        return self[action] == 'no-op'

    def is_door_usage(self, action: Union[str, int]):
        if isinstance(action, str):
            action = self.by_name(action)
        return self[action] == 'use_door'


class StateSlices(Register):

    @property
    def AGENTSTARTIDX(self):
        if self._agent_start_idx:
            return self._agent_start_idx
        else:
            self._agent_start_idx = min([idx for idx, x in self.items() if 'agent' in x])
            return self._agent_start_idx

    def __init__(self):
        super(StateSlices, self).__init__()
        self._agent_start_idx = None


class Zones(Register):

    @property
    def danger_zone(self):
        return self._zone_slices[self.by_name(h.DANGER_ZONE)]

    @property
    def accounting_zones(self):
        return [self[idx] for idx, name in self.items() if name != h.DANGER_ZONE]

    def __init__(self, parsed_level):
        super(Zones, self).__init__()
        slices = list()
        self._accounting_zones = list()
        self._danger_zones = list()
        for symbol in np.unique(parsed_level):
            if symbol == h.WALL:
                continue
            elif symbol == h.DANGER_ZONE:
                self + symbol
                slices.append(h.one_hot_level(parsed_level, symbol))
                self._danger_zones.append(symbol)
            else:
                self + symbol
                slices.append(h.one_hot_level(parsed_level, symbol))
                self._accounting_zones.append(symbol)

        self._zone_slices = np.stack(slices)

    def __getitem__(self, item):
        return self._zone_slices[item]

    def get_name(self, item):
        return self._register[item]

    def by_name(self, item):
        return self[super(Zones, self).by_name(item)]

    def register_additional_items(self, other: Union[str, List[str]]):
        raise AttributeError('You are not allowed to add additional Zones in runtime.')
