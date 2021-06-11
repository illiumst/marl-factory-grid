from typing import Union, List, NamedTuple
import numpy as np

from environments import helpers as h


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


class AgentState:

    def __init__(self, i: int, action: int):
        self.i = i
        self.action = action

        self.collision_vector = None
        self.action_valid = None
        self.pos = None
        self.info = {}

    @property
    def collisions(self):
        return np.argwhere(self.collision_vector != 0).flatten()

    def update(self, **kwargs):                             # is this hacky?? o.0
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                raise AttributeError(f'"{key}" cannot be updated, this attr is not a part of {self.__class__.__name__}')


class Register:

    @property
    def n(self):
        return len(self)

    def __init__(self):
        self._register = dict()

    def __len__(self):
        return len(self._register)

    def __add__(self, other: Union[str, List[str]]):
        other = other if isinstance(other, list) else [other]
        assert all([isinstance(x, str) for x in other]), f'All item names have to be of type {str}.'
        self._register.update({key+len(self._register): value for key, value in enumerate(other)})
        return self

    def register_additional_items(self, other: Union[str, List[str]]):
        self_with_additional_items = self + other
        return self_with_additional_items

    def keys(self):
        return self._register.keys()

    def items(self):
        return self._register.items()

    def __getitem__(self, item):
        return self._register[item]

    def by_name(self, item):
        return list(self._register.keys())[list(self._register.values()).index(item)]

    def __repr__(self):
        return f'{self.__class__.__name__}({self._register})'


class Actions(Register):

    @property
    def movement_actions(self):
        return self._movement_actions

    def __init__(self, movement_properties: MovementProperties):
        self.allow_no_op = movement_properties.allow_no_op
        self.allow_diagonal_movement = movement_properties.allow_diagonal_movement
        self.allow_square_movement = movement_properties.allow_square_movement
        # FIXME: There is a bug in helpers because there actions are ints. and the order matters.
        # assert not(self.allow_square_movement is False and self.allow_diagonal_movement is True), \
        #     "There is a bug in helpers!!!"
        super(Actions, self).__init__()

        if self.allow_square_movement:
            self + ['north', 'east', 'south', 'west']
        if self.allow_diagonal_movement:
            self + ['north_east', 'south_east', 'south_west', 'north_west']
        self._movement_actions = self._register.copy()
        if self.allow_no_op:
            self + 'no-op'

    def is_moving_action(self, action: Union[str, int]):
        if isinstance(action, str):
            return action in self.movement_actions.values()
        else:
            return self[action] in self.movement_actions.values()

    def is_no_op(self, action: Union[str, int]):
        if isinstance(action, str):
            action = self.by_name(action)
        return self[action] == 'no-op'


class StateSlice(Register):

    def __init__(self, n_agents: int):
        super(StateSlice, self).__init__()
        offset = 1  # AGENT_START_IDX
        self.register_additional_items(['level', *[f'agent#{i}' for i in range(offset, n_agents+offset)]])


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
