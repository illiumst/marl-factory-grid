import itertools
import random
from enum import Enum
from typing import List, Union

import networkx as nx
import numpy as np

from environments.factory.base.objects import Entity, Tile, Agent, Door, Slice, Action
from environments.utility_classes import MovementProperties
from environments import helpers as h
from environments.helpers import Constants as c


class Register:
    _accepted_objects = Entity

    @classmethod
    def from_argwhere_coordinates(cls, positions: (int, int), tiles):
        entities = [cls._accepted_objects(i, tiles.by_pos(position)) for i, position in enumerate(positions)]
        registered_obj = cls()
        registered_obj.register_additional_items(entities)
        return registered_obj

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def n(self):
        return len(self)

    def __init__(self):
        self._register = dict()
        self._names = dict()

    def __len__(self):
        return len(self._register)

    def __iter__(self):
        return iter(self.values())

    def __add__(self, other: _accepted_objects):
        assert isinstance(other, self._accepted_objects), f'All item names have to be of type ' \
                                                          f'{self._accepted_objects}, ' \
                                                          f'but were {other.__class__}.,'
        self._names.update({other.name: len(self._register)})
        self._register.update({len(self._register): other})
        return self

    def register_additional_items(self, others: List[_accepted_objects]):
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
        return self[self._names[item]]

    def by_enum(self, enum: Enum):
        return self[self._names[enum.name]]

    def __repr__(self):
        return f'{self.__class__.__name__}({self._register})'

    def get_name(self, item):
        return self._register[item].name

    def get_idx_by_name(self, item):
        return self._names[item]

    def get_idx(self, enum: Enum):
        return self._names[enum.name]

    @classmethod
    def from_tiles(cls, tiles, **kwargs):
        entities = [cls._accepted_objects(f'{cls._accepted_objects.__name__.upper()}#{i}', tile, **kwargs)
                    for i, tile in enumerate(tiles)]
        registered_obj = cls()
        registered_obj.register_additional_items(entities)
        return registered_obj


class EntityRegister(Register):

    @classmethod
    def from_argwhere_coordinates(cls, argwhere_coordinates):
        tiles = cls()
        tiles.register_additional_items([cls._accepted_objects(i, pos) for i, pos in enumerate(argwhere_coordinates)])
        return tiles

    def __init__(self):
        super(EntityRegister, self).__init__()
        self._tiles = dict()

    def __add__(self, other):
        super(EntityRegister, self).__add__(other)
        self._tiles[other.pos] = other

    def by_pos(self, pos):
        if isinstance(pos, np.ndarray):
            pos = tuple(pos)
        try:
            return self._tiles[pos]
        except KeyError:
            return None


class Entities(Register):

    _accepted_objects = Register

    def __init__(self):
        super(Entities, self).__init__()

    def __iter__(self):
        return iter([x for sublist in self.values() for x in sublist])

    @classmethod
    def from_argwhere_coordinates(cls, positions):
        raise AttributeError()


class FloorTiles(EntityRegister):
    _accepted_objects = Tile

    @property
    def occupied_tiles(self):
        tiles = [tile for tile in self if tile.is_occupied()]
        random.shuffle(tiles)
        return tiles

    @property
    def empty_tiles(self):
        tiles = [tile for tile in self if tile.is_empty()]
        random.shuffle(tiles)
        return tiles


class Agents(Register):

    _accepted_objects = Agent

    @property
    def positions(self):
        return [agent.pos for agent in self]


class Doors(EntityRegister):
    _accepted_objects = Door

    def tick_doors(self):
        for door in self:
            door.tick()


class Actions(Register):

    _accepted_objects = Action

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
            self.register_additional_items([self._accepted_objects(direction) for direction in h.MANHATTAN_MOVES])
        if self.allow_diagonal_movement:
            self.register_additional_items([self._accepted_objects(direction) for direction in h.DIAGONAL_MOVES])
        self._movement_actions = self._register.copy()
        if self.can_use_doors:
            self.register_additional_items([self._accepted_objects('use_door')])
        if self.allow_no_op:
            self.register_additional_items([self._accepted_objects('no-op')])

    def is_moving_action(self, action: Union[int]):
        #if isinstance(action, Action):
        #    return (action.name in h.MANHATTAN_MOVES and self.allow_square_movement) or \
        #           (action.name in h.DIAGONAL_MOVES and self.allow_diagonal_movement)
        #else:
        return action in self.movement_actions.keys()

    def is_no_op(self, action: Union[str, int]):
        if isinstance(action, str):
            action = self.by_name(action)
        return self[action].name == 'no-op'

    def is_door_usage(self, action: Union[str, int]):
        if isinstance(action, str):
            action = self.by_name(action)
        return self[action].name == 'use_door'


class StateSlices(Register):

    _accepted_objects = Slice

    @property
    def AGENTSTARTIDX(self):
        if self._agent_start_idx:
            return self._agent_start_idx
        else:
            self._agent_start_idx = min([idx for idx, x in self.items() if c.AGENT.name in x.name])
            return self._agent_start_idx

    def __init__(self):
        super(StateSlices, self).__init__()
        self._agent_start_idx = None

    def _gather_occupation(self, excluded_slices):
        exclusion = excluded_slices or []
        assert isinstance(exclusion, (int, list))
        exclusion = exclusion if isinstance(exclusion, list) else [exclusion]

        result = np.sum([x for i, x in self.items() if i not in exclusion], axis=0)
        return result

    def free_cells(self, excluded_slices: Union[None, List[int], int] = None) -> np.array:
        occupation = self._gather_occupation(excluded_slices)
        free_cells = np.argwhere(occupation == c.IS_FREE_CELL)
        np.random.shuffle(free_cells)
        return free_cells

    def occupied_cells(self, excluded_slices: Union[None, List[int], int] = None) -> np.array:
        occupation = self._gather_occupation(excluded_slices)
        occupied_cells = np.argwhere(occupation == c.IS_OCCUPIED_CELL.value)
        np.random.shuffle(occupied_cells)
        return occupied_cells


class Zones(Register):

    @property
    def danger_zone(self):
        return self._zone_slices[self.by_enum(c.DANGER_ZONE)]

    @property
    def accounting_zones(self):
        return [self[idx] for idx, name in self.items() if name != c.DANGER_ZONE.value]

    def __init__(self, parsed_level):
        raise NotImplementedError('This needs a Rework')
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