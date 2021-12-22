import numbers
import random
from abc import ABC
from typing import List, Union, Dict, Tuple

import numpy as np

from environments.factory.base.objects import Entity, Tile, Agent, Door, Action, Wall, PlaceHolder, GlobalPosition, \
    Object, EnvObject
from environments.utility_classes import MovementProperties
from environments import helpers as h
from environments.helpers import Constants as c

##########################################################################
# ##################### Base Register Definition ####################### #
##########################################################################


class ObjectRegister:
    _accepted_objects = Object

    @property
    def name(self):
        return f'{self.__class__.__name__}'

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

    def _get_index(self, item):
        try:
            return next(i for i, v in enumerate(self._register.values()) if v == item)
        except (StopIteration, AssertionError):
            return None

    def __getitem__(self, item):
        if isinstance(item, (int, np.int64, np.int32)):
            if item < 0:
                item = len(self._register) - abs(item)
            try:
                return next(v for i, v in enumerate(self._register.values()) if i == item)
            except StopIteration:
                return None
        try:
            return self._register[item]
        except KeyError:
            return None

    def __repr__(self):
        return f'{self.__class__.__name__}({self._register})'


class EnvObjectRegister(ObjectRegister):

    _accepted_objects = EnvObject

    def __init__(self, obs_shape: (int, int), *args, **kwargs):
        super(EnvObjectRegister, self).__init__(*args, **kwargs)
        self._shape = obs_shape
        self._array = None
        self.hide_from_obs_builder = False
        self._lazy_eval_transforms = []

    def register_item(self, other: EnvObject):
        super(EnvObjectRegister, self).register_item(other)
        if self._array is None:
            self._array = np.zeros((1, *self._shape))
        self.notify_change_to_value(other)

    def as_array(self):
        if self._lazy_eval_transforms:
            idxs, values  = zip(*self._lazy_eval_transforms)
            # nuumpy put repects the ordering so that
            np.put(self._array, idxs, values)
            self._lazy_eval_transforms = []
        return self._array

    def summarize_states(self, n_steps=None):
        return [val.summarize_state(n_steps=n_steps) for val in self.values()]

    def notify_change_to_free(self, env_object: EnvObject):
        self._array_change_notifyer(env_object, value=c.FREE_CELL.value)

    def notify_change_to_value(self, env_object: EnvObject):
        self._array_change_notifyer(env_object)

    def _array_change_notifyer(self, env_object: EnvObject, value=None):
        pos = self._get_index(env_object)
        value = value if value is not None else env_object.encoding
        self._lazy_eval_transforms.append((pos, value))

    def __delitem__(self, name):
        self.notify_change_to_free(self._register[name])
        del self._register[name]

    def delete_env_object(self, env_object: EnvObject):
        del self[env_object.name]

    def delete_env_object_by_name(self, name):
        del self[name]


class EntityRegister(EnvObjectRegister, ABC):

    _accepted_objects = Entity

    @classmethod
    def from_tiles(cls, tiles, *args, entity_kwargs=None, **kwargs):
        # objects_name = cls._accepted_objects.__name__
        register_obj = cls(*args, **kwargs)
        entities = [cls._accepted_objects(tile, register_obj, str_ident=i,
                                          **entity_kwargs if entity_kwargs is not None else {})
                    for i, tile in enumerate(tiles)]
        register_obj.register_additional_items(entities)
        return register_obj

    @classmethod
    def from_argwhere_coordinates(cls, positions: [(int, int)], tiles, *args, entity_kwargs=None, **kwargs, ):
        return cls.from_tiles([tiles.by_pos(position) for position in positions], *args, entity_kwargs=entity_kwargs,
                              **kwargs)

    @property
    def positions(self):
        return [x.pos for x in self]

    @property
    def tiles(self):
        return [entity.tile for entity in self]

    @property
    def encodings(self):
        return [x.encoding for x in self]

    def __init__(self, level_shape, *args,
                 is_blocking_light: bool = False,
                 can_be_shadowed: bool = True,
                 individual_slices: bool = False, **kwargs):
        super(EntityRegister, self).__init__(level_shape, *args, **kwargs)
        self._lazy_eval_transforms = []
        self.can_be_shadowed = can_be_shadowed
        self.individual_slices = individual_slices
        self.is_blocking_light = is_blocking_light

    def __delitem__(self, name):
        idx, obj = next((i, obj) for i, obj in enumerate(self) if obj.name == name)
        obj.tile.leave(obj)
        super(EntityRegister, self).__delitem__(name)
        if self.individual_slices:
            self._array = np.delete(self._array, idx, axis=0)

    def as_array(self):
        if self._lazy_eval_transforms:
            idxs, values  = zip(*self._lazy_eval_transforms)
            # numpy put repects the ordering so that
            # Todo: Export the index building in a seperate function
            np.put(self._array, [np.ravel_multi_index(idx, self._array.shape) for idx in idxs], values)
            self._lazy_eval_transforms = []
        return self._array

    def _array_change_notifyer(self, entity, pos=None, value=None):
        # Todo: Export the contruction in a seperate function
        pos = pos if pos is not None else entity.pos
        value = value if value is not None else entity.encoding
        x, y = pos
        if self.individual_slices:
            idx = (self._get_index(entity), x, y)
        else:
            idx = (0, x, y)
        self._lazy_eval_transforms.append((idx, value))

    def by_pos(self, pos: Tuple[int, int]):
        try:
            return next(item for item in self if item.pos == tuple(pos))
        except StopIteration:
            return None


class BoundRegisterMixin(EnvObjectRegister, ABC):

    @classmethod
    def from_entities_to_bind(self, entitites):
        def from_values(cls, values: Union[str, numbers.Number, List[Union[str, numbers.Number]]],
                        *args, object_kwargs=None, **kwargs):
            # objects_name = cls._accepted_objects.__name__
            if isinstance(values, (str, numbers.Number)):
                values = [values]
            register_obj = cls(*args, **kwargs)
            objects = [cls._accepted_objects(register_obj, str_ident=i, fill_value=value,
                                             **object_kwargs if object_kwargs is not None else {})
                       for i, value in enumerate(values)]
            register_obj.register_additional_items(objects)
            return register_obj


class MovingEntityObjectRegister(EntityRegister, ABC):

    def __init__(self, *args, **kwargs):
        super(MovingEntityObjectRegister, self).__init__(*args, **kwargs)

    def notify_change_to_value(self, entity):
        super(MovingEntityObjectRegister, self).notify_change_to_value(entity)
        if entity.last_pos != c.NO_POS.value:
            try:
                self._array_change_notifyer(entity, entity.last_pos, value=c.FREE_CELL.value)
            except AttributeError:
                pass


##########################################################################
# ################# Objects and Entity Registers ####################### #
##########################################################################


class GlobalPositions(EnvObjectRegister):
    _accepted_objects = GlobalPosition
    is_blocking_light = False
    can_be_shadowed = False
    hide_from_obs_builder = True

    def __init__(self, *args, **kwargs):
        super(GlobalPositions, self).__init__(*args, is_per_agent=True, individual_slices=True, **kwargs)

    def as_array(self):
        # Todo make this lazy?
        return np.stack([gp.as_array() for inv_idx, gp in enumerate(self)])

    def spawn_GlobalPositionObjects(self, obs_shape, agents):
        global_positions = [self._accepted_objects(self._shape, obs_shape, agent)
                            for _, agent in enumerate(agents)]
        # noinspection PyTypeChecker
        self.register_additional_items(global_positions)

    def summarize_states(self, n_steps=None):
        return {}

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, inv in enumerate(self) if inv.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def by_entity(self, entity):
        try:
            return next((inv for inv in self if inv.belongs_to_entity(entity)))
        except StopIteration:
            return None


class PlaceHolders(EnvObjectRegister):
    _accepted_objects = PlaceHolder

    def __init__(self, *args, **kwargs):
        assert not 'individual_slices' in kwargs, 'Keyword - "individual_slices": "True" and must not be altered'
        kwargs.update(individual_slices=False)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_values(cls, values: Union[str, numbers.Number, List[Union[str, numbers.Number]]],
                    *args, object_kwargs=None, **kwargs):
        # objects_name = cls._accepted_objects.__name__
        if isinstance(values, (str, numbers.Number)):
            values = [values]
        register_obj = cls(*args, **kwargs)
        objects = [cls._accepted_objects(register_obj, str_ident=i, fill_value=value,
                                         **object_kwargs if object_kwargs is not None else {})
                   for i, value in enumerate(values)]
        register_obj.register_additional_items(objects)
        return register_obj

    # noinspection DuplicatedCode
    def as_array(self):
        for idx, placeholder in enumerate(self):
            if isinstance(placeholder.encoding, numbers.Number):
                self._array[idx][:] = placeholder.fill_value
            elif isinstance(placeholder.fill_value, str):
                if placeholder.fill_value.lower() in ['normal', 'n']:
                    self._array[:] = np.random.normal(size=self._array.shape)
                else:
                    raise ValueError('Choose one of: ["normal", "N"]')
            else:
                raise TypeError('Objects of type "str" or "number" is required here.')

        return self._array


class Entities(ObjectRegister):
    _accepted_objects = EntityRegister

    @property
    def arrays(self):
        return {key: val.as_array() for key, val in self.items()}

    @property
    def obs_arrays(self):
        return {key: val.as_array() for key, val in self.items() if not val.hide_from_obs_builder}

    @property
    def names(self):
        return list(self._register.keys())

    def __init__(self):
        super(Entities, self).__init__()

    def iter_individual_entitites(self):
        return iter((x for sublist in self.values() for x in sublist))

    def register_item(self, other: dict):
        assert not any([key for key in other.keys() if key in self.keys()]), \
            "This group of entities has already been registered!"
        self._register.update(other)
        return self

    def register_additional_items(self, others: Dict):
        return self.register_item(others)

    def by_pos(self, pos: (int, int)):
        found_entities = [y for y in (x.by_pos(pos) for x in self.values() if hasattr(x, 'by_pos')) if y is not None]
        return found_entities


class WallTiles(EntityRegister):
    _accepted_objects = Wall
    _light_blocking = True
    hide_from_obs_builder = True

    def as_array(self):
        if not np.any(self._array):
            # Which is Faster?
            # indices = [x.pos for x in self]
            # np.put(self._array, [np.ravel_multi_index((0, *x), self._array.shape) for x in indices], self.encodings)
            x, y = zip(*[x.pos for x in self])
            self._array[0, x, y] = self.encoding
        return self._array

    def __init__(self, *args, **kwargs):
        super(WallTiles, self).__init__(*args, is_blocking_light=self._light_blocking, individual_slices=False,
                                        **kwargs)

    @property
    def encoding(self):
        return c.OCCUPIED_CELL.value

    @classmethod
    def from_argwhere_coordinates(cls, argwhere_coordinates, *args, **kwargs):
        tiles = cls(*args, **kwargs)
        # noinspection PyTypeChecker
        tiles.register_additional_items(
            [cls._accepted_objects(pos, tiles, is_blocking_light=cls._light_blocking)
             for pos in argwhere_coordinates]
        )
        return tiles

    @classmethod
    def from_tiles(cls, tiles, *args, **kwargs):
        raise RuntimeError()

    def summarize_states(self, n_steps=None):
        if n_steps == h.STEPS_START:
            return super(WallTiles, self).summarize_states(n_steps=n_steps)
        else:
            return {}


class FloorTiles(WallTiles):
    _accepted_objects = Tile
    _light_blocking = False

    def __init__(self, *args, **kwargs):
        super(FloorTiles, self).__init__(*args, **kwargs)

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

    def summarize_states(self, n_steps=None):
        # Do not summarize
        return {}


class Agents(MovingEntityObjectRegister):
    _accepted_objects = Agent

    def __init__(self, *args, hide_from_obs_builder=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hide_from_obs_builder = hide_from_obs_builder

    @DeprecationWarning
    def Xas_array(self):
        # Super Safe Version
        # self._array[:] = c.FREE_CELL.value
        indices = list(zip(range(len(self)), *zip(*[x.last_pos for x in self])))
        np.put(self._array, [np.ravel_multi_index(x, self._array.shape) for x in indices], c.FREE_CELL.value)
        indices = list(zip(range(len(self)), *zip(*[x.pos for x in self])))
        np.put(self._array, [np.ravel_multi_index(x, self._array.shape) for x in indices], self.encodings)

        if self.individual_slices:
            return self._array
        else:
            return self._array.sum(axis=0, keepdims=True)

    @property
    def positions(self):
        return [agent.pos for agent in self]

    def replace_agent(self, key, agent):
        old_agent = self[key]
        self[key].tile.leave(self[key])
        agent._name = old_agent.name
        self._register[agent.name] = agent


class Doors(EntityRegister):

    def __init__(self, *args, **kwargs):
        super(Doors, self).__init__(*args, is_blocking_light=True, **kwargs)

    _accepted_objects = Door

    def get_near_position(self, position: (int, int)) -> Union[None, Door]:
        try:
            return next(door for door in self if position in door.access_area)
        except StopIteration:
            return None

    def tick_doors(self):
        for door in self:
            door.tick()


class Actions(ObjectRegister):
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
                                            for direction in h.MovingAction.square()])
        if self.allow_diagonal_movement:
            self.register_additional_items([self._accepted_objects(enum_ident=direction)
                                            for direction in h.MovingAction.diagonal()])
        self._movement_actions = self._register.copy()
        if self.can_use_doors:
            self.register_additional_items([self._accepted_objects(enum_ident=h.EnvActions.USE_DOOR)])
        if self.allow_no_op:
            self.register_additional_items([self._accepted_objects(enum_ident=h.EnvActions.NOOP)])

    def is_moving_action(self, action: Union[int]):
        return action in self.movement_actions.values()


class Zones(ObjectRegister):

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
