import random
from abc import ABC
from typing import List, Union, Dict

import numpy as np

from environments.factory.base.objects import Entity, Tile, Agent, Door, Action, Wall
from environments.utility_classes import MovementProperties
from environments import helpers as h
from environments.helpers import Constants as c


class Register:
    _accepted_objects = Entity

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self, *args, **kwargs):
        self._register = dict()

    def __len__(self):
        return len(self._register)

    def __iter__(self):
        return iter(self.values())

    def register_item(self, other: _accepted_objects):
        assert isinstance(other, self._accepted_objects), f'All item names have to be of type ' \
                                                          f'{self._accepted_objects}, ' \
                                                          f'but were {other.__class__}.,'
        self._register.update({other.name: other})
        return self

    def register_additional_items(self, others: List[_accepted_objects]):
        for other in others:
            self.register_item(other)
        return self

    def keys(self):
        return self._register.keys()

    def values(self):
        return self._register.values()

    def items(self):
        return self._register.items()

    def __getitem__(self, item):
        if isinstance(item, int):
            try:
                return next(v for i, v in enumerate(self._register.values()) if i == item)
            except StopIteration:
                return None
        return self._register[item]

    def __repr__(self):
        return f'{self.__class__.__name__}({self._register})'


class ObjectRegister(Register):
    def __init__(self, level_shape: (int, int), *args, individual_slices=False, is_per_agent=False, **kwargs):
        super(ObjectRegister, self).__init__(*args, **kwargs)
        self.is_per_agent = is_per_agent
        self.individual_slices = individual_slices
        self._level_shape = level_shape
        self._array = None

    def register_item(self, other):
        super(ObjectRegister, self).register_item(other)
        if self._array is None:
            self._array = np.zeros((1, *self._level_shape))
        else:
            if self.individual_slices:
                self._array = np.concatenate((self._array, np.zeros((1, *self._level_shape))))


class EntityObjectRegister(ObjectRegister, ABC):

    def as_array(self):
        raise NotImplementedError

    @classmethod
    def from_tiles(cls, tiles, *args, **kwargs):
        # objects_name = cls._accepted_objects.__name__
        entities = [cls._accepted_objects(tile, **kwargs)
                    for tile in tiles]
        register_obj = cls(*args)
        register_obj.register_additional_items(entities)
        return register_obj

    @classmethod
    def from_argwhere_coordinates(cls, positions: [(int, int)], tiles, *args, **kwargs):
        return cls.from_tiles([tiles.by_pos(position) for position in positions], *args, **kwargs)

    @property
    def positions(self):
        return [x.pos for x in self]

    @property
    def tiles(self):
        return [entity.tile for entity in self]

    def __init__(self, *args, is_blocking_light=False, is_observable=True, can_be_shadowed=True, **kwargs):
        super(EntityObjectRegister, self).__init__(*args, **kwargs)
        self.can_be_shadowed = can_be_shadowed
        self.is_blocking_light = is_blocking_light
        self.is_observable = is_observable

    def by_pos(self, pos):
        if isinstance(pos, np.ndarray):
            pos = tuple(pos)
        try:
            return next(item for item in self.values() if item.pos == pos)
        except StopIteration:
            return None


class MovingEntityObjectRegister(EntityObjectRegister, ABC):

    def __init__(self, *args, **kwargs):
        super(MovingEntityObjectRegister, self).__init__(*args, **kwargs)

    def by_pos(self, pos):
        if isinstance(pos, np.ndarray):
            pos = tuple(pos)
        try:
            return next(x for x in self if x.pos == pos)
        except StopIteration:
            return None

    def delete_item(self, item):
        if not isinstance(item, str):
            item = item.name
        del self._register[item]


class Entities(Register):

    _accepted_objects = EntityObjectRegister

    @property
    def arrays(self):
        return {key: val.as_array() for key, val in self.items() if val.is_observable}

    @property
    def names(self):
        return list(self._register.keys())

    def __init__(self):
        super(Entities, self).__init__()

    def __iter__(self):
        return iter([x for sublist in self.values() for x in sublist])

    def register_item(self, other: dict):
        assert not any([key for key in other.keys() if key in self.keys()]), \
            "This group of entities has already been registered!"
        self._register.update(other)
        return self

    def register_additional_items(self, others: Dict):
        return self.register_item(others)


class WallTiles(EntityObjectRegister):
    _accepted_objects = Wall
    _light_blocking = True

    def as_array(self):
        if not np.any(self._array):
            x, y = zip(*[x.pos for x in self])
            self._array[0, x, y] = self.encoding
        return self._array

    def __init__(self, *args, **kwargs):
        super(WallTiles, self).__init__(*args, individual_slices=False,
                                        is_blocking_light=self._light_blocking, **kwargs)

    @property
    def encoding(self):
        return c.OCCUPIED_CELL.value

    @property
    def array(self):
        return self._array

    @classmethod
    def from_argwhere_coordinates(cls, argwhere_coordinates, *args, **kwargs):
        tiles = cls(*args, **kwargs)
        # noinspection PyTypeChecker
        tiles.register_additional_items(
            [cls._accepted_objects(pos, is_blocking_light=cls._light_blocking)
             for pos in argwhere_coordinates]
        )
        return tiles

    @classmethod
    def from_tiles(cls, tiles, *args, **kwargs):
        raise RuntimeError()


class FloorTiles(WallTiles):

    _accepted_objects = Tile
    _light_blocking = False

    def __init__(self, *args, **kwargs):
        super(FloorTiles, self).__init__(*args, is_observable=False, **kwargs)

    @property
    def encoding(self):
        return c.FREE_CELL.value

    @property
    def occupied_tiles(self):
        tiles = [tile for tile in self if tile.is_occupied()]
        random.shuffle(tiles)
        return tiles

    @property
    def empty_tiles(self) -> List[Tile]:
        tiles = [tile for tile in self if tile.is_empty()]
        random.shuffle(tiles)
        return tiles

    @classmethod
    def from_tiles(cls, tiles, *args, **kwargs):
        raise RuntimeError()


class Agents(MovingEntityObjectRegister):

    def as_array(self):
        self._array[:] = c.FREE_CELL.value
        # noinspection PyTupleAssignmentBalance
        for z, x, y, v in zip(range(len(self)), *zip(*[x.pos for x in self]), [x.encoding for x in self]):
            if self.individual_slices:
                self._array[z, x, y] += v
            else:
                self._array[0, x, y] += v
        if self.individual_slices:
            return self._array
        else:
            return self._array.sum(axis=0, keepdims=True)

    _accepted_objects = Agent

    @property
    def positions(self):
        return [agent.pos for agent in self]


class Doors(EntityObjectRegister):

    def __init__(self, *args, **kwargs):
        super(Doors, self).__init__(*args, is_blocking_light=True, **kwargs)

    def as_array(self):
        self._array[:] = 0
        for door in self:
            self._array[0, door.x, door.y] = door.encoding
        return self._array

    _accepted_objects = Door

    def get_near_position(self, position: (int, int)) -> Union[None, Door]:
        try:
            return next(door for door in self if position in door.access_area)
        except StopIteration:
            return None

    def tick_doors(self):
        for door in self:
            door.tick()


class Actions(Register):

    _accepted_objects = Action

    @property
    def movement_actions(self):
        return self._movement_actions

    # noinspection PyTypeChecker
    def __init__(self, movement_properties: MovementProperties, can_use_doors=False):
        self.allow_no_op = movement_properties.allow_no_op
        self.allow_diagonal_movement = movement_properties.allow_diagonal_movement
        self.allow_square_movement = movement_properties.allow_square_movement
        self.can_use_doors = can_use_doors
        super(Actions, self).__init__()

        if self.allow_square_movement:
            self.register_additional_items([self._accepted_objects(enum_ident=direction)
                                            for direction in h.ManhattanMoves])
        if self.allow_diagonal_movement:
            self.register_additional_items([self._accepted_objects(enum_ident=direction)
                                            for direction in h.DiagonalMoves])
        self._movement_actions = self._register.copy()
        if self.can_use_doors:
            self.register_additional_items([self._accepted_objects(enum_ident=h.EnvActions.USE_DOOR)])
        if self.allow_no_op:
            self.register_additional_items([self._accepted_objects(enum_ident=h.EnvActions.NOOP)])

    def is_moving_action(self, action: Union[int]):
        return action in self.movement_actions.values()


class Zones(Register):

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
            if symbol == c.WALL.value:
                continue
            elif symbol == c.DANGER_ZONE.value:
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

    def register_additional_items(self, other: Union[str, List[str]]):
        raise AttributeError('You are not allowed to add additional Zones in runtime.')
