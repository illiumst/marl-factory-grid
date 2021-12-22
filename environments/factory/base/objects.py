from collections import defaultdict
from enum import Enum
from typing import Union

import networkx as nx
import numpy as np

from environments import helpers as h
from environments.helpers import Constants as c
import itertools

##########################################################################
# ##################### Base Object Definition ######################### #
##########################################################################


class Object:

    """Generell Objects for Organisation and Maintanance such as Actions etc..."""

    _u_idx = defaultdict(lambda: 0)

    def __bool__(self):
        return True

    @property
    def name(self):
        return self._name

    @property
    def identifier(self):
        if self._enum_ident is not None:
            return self._enum_ident
        elif self._str_ident is not None:
            return self._str_ident
        else:
            return self._name

    def __init__(self, str_ident: Union[str, None] = None, enum_ident: Union[Enum, None] = None,
                 is_blocking_light=False, **kwargs):

        self._str_ident = str_ident
        self._enum_ident = enum_ident

        if self._enum_ident is not None and self._str_ident is None:
            self._name = f'{self.__class__.__name__}[{self._enum_ident.name}]'
        elif self._str_ident is not None and self._enum_ident is None:
            self._name = f'{self.__class__.__name__}[{self._str_ident}]'
        elif self._str_ident is None and self._enum_ident is None:
            self._name = f'{self.__class__.__name__}#{Object._u_idx[self.__class__.__name__]}'
            Object._u_idx[self.__class__.__name__] += 1
        else:
            raise ValueError('Please use either of the idents.')

        self._is_blocking_light = is_blocking_light
        if kwargs:
            print(f'Following kwargs were passed, but ignored: {kwargs}')

    def __repr__(self):
        return f'{self.name}'

    def __eq__(self, other) -> bool:
        if self._enum_ident is not None:
            if isinstance(other, Enum):
                return other == self._enum_ident
            elif isinstance(other, Object):
                return other._enum_ident == self._enum_ident
            else:
                raise ValueError('Must be evaluated against an Enunm Identifier or Object with such.')
        else:
            assert isinstance(other, Object), ' This Object can only be compared to other Objects.'
            return other.name == self.name


class EnvObject(Object):

    """Objects that hold Information that are observable, but have no position on the env grid. Inventories etc..."""

    _u_idx = defaultdict(lambda: 0)

    @property
    def encoding(self):
        return c.OCCUPIED_CELL.value

    def __init__(self, register, **kwargs):
        super(EnvObject, self).__init__(**kwargs)
        self._register = register


class BoundingMixin:

    @property
    def bound_entity(self):
        return self._bound_entity

    def __init__(self, entity_to_be_bound, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert entity_to_be_bound is not None
        self._bound_entity = entity_to_be_bound

    def __repr__(self):
        s = super(BoundingMixin, self).__repr__()
        i = s[:s.find('(')]
        return f'{s[:i]}[{self.bound_entity.name}]{s[i:]}'

    @property
    def name(self):
        return f'{super(BoundingMixin, self).name}({self._bound_entity.name})'

    def belongs_to_entity(self, entity):
        return entity == self.bound_entity


class Entity(EnvObject):
    """Full Env Entity that lives on the env Grid. Doors, Items, Dirt etc..."""

    @property
    def is_blocking_light(self):
        return self._is_blocking_light

    @property
    def can_collide(self):
        return True

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def pos(self):
        return self._tile.pos

    @property
    def tile(self):
        return self._tile

    def __init__(self, tile, *args, is_blocking_light=True,  **kwargs):
        super().__init__(*args, **kwargs)
        self._tile = tile
        self._is_blocking_light = is_blocking_light
        tile.enter(self)

    def summarize_state(self, **_) -> dict:
        return dict(name=str(self.name), x=int(self.x), y=int(self.y),
                    tile=str(self.tile.name), can_collide=bool(self.can_collide))

    def __repr__(self):
        return super(Entity, self).__repr__() + f'(@{self.pos})'


class MoveableEntity(Entity):

    @property
    def last_tile(self):
        return self._last_tile

    @property
    def last_pos(self):
        if self._last_tile:
            return self._last_tile.pos
        else:
            return c.NO_POS.value

    @property
    def direction_of_view(self):
        last_x, last_y = self.last_pos
        curr_x, curr_y = self.pos
        return last_x-curr_x, last_y-curr_y

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_tile = None

    def move(self, next_tile):
        curr_tile = self.tile
        if curr_tile != next_tile:
            next_tile.enter(self)
            curr_tile.leave(self)
            self._tile = next_tile
            self._last_tile = curr_tile
            self._register.notify_change_to_value(self)
            return True
        else:
            return False


##########################################################################
# ####################### Objects and Entitys ########################## #
##########################################################################


class Action(Object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PlaceHolder(Object):

    def __init__(self, *args, fill_value=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._fill_value = fill_value

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        return self._fill_value

    @property
    def name(self):
        return "PlaceHolder"


class GlobalPosition(EnvObject):

    def belongs_to_entity(self, entity):
        return self._agent == entity

    def __init__(self, level_shape, obs_shape, agent, normalized: bool = True):
        super(GlobalPosition, self).__init__(self)
        self._obs_shape = (1, *obs_shape) if len(obs_shape) == 2 else obs_shape
        self._agent = agent
        self._level_shape = level_shape
        self._normalized = normalized

    def as_array(self):
        pos_array = np.zeros(self._obs_shape)
        for xy in range(1):
            pos_array[0, 0, xy] = self._agent.pos[xy] / self._level_shape[xy]
        return pos_array


class Tile(EnvObject):

    @property
    def encoding(self):
        return c.FREE_CELL.value

    @property
    def guests_that_can_collide(self):
        return [x for x in self.guests if x.can_collide]

    @property
    def guests(self):
        return self._guests.values()

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def pos(self):
        return self._pos

    def __init__(self, pos, *args, **kwargs):
        super(Tile, self).__init__(*args, **kwargs)
        self._guests = dict()
        self._pos = tuple(pos)

    def __len__(self):
        return len(self._guests)

    def is_empty(self):
        return not len(self._guests)

    def is_occupied(self):
        return bool(len(self._guests))

    def enter(self, guest):
        if guest.name not in self._guests:
            self._guests.update({guest.name: guest})
            return True
        else:
            return False

    def leave(self, guest):
        try:
            del self._guests[guest.name]
        except (ValueError, KeyError):
            return False
        return True

    def __repr__(self):
        return f'{self.name}(@{self.pos})'

    def summarize_state(self, **_):
        return dict(name=self.name, x=int(self.x), y=int(self.y))


class Wall(Tile):

    @property
    def encoding(self):
        return c.OCCUPIED_CELL.value

    pass


class Door(Entity):

    @property
    def can_collide(self):
        if self.has_area:
            return False if self.is_open else True
        else:
            return False

    @property
    def encoding(self):
        # This is important as it shadow is checked by occupation value
        return c.OCCUPIED_CELL.value if self.is_closed else 2

    @property
    def str_state(self):
        return 'open' if self.is_open else 'closed'

    @property
    def access_area(self):
        return [node for node in self.connectivity.nodes
                if node not in range(len(self.connectivity_subgroups)) and node != self.pos]

    def __init__(self, *args, context, closed_on_init=True, auto_close_interval=10, has_area=False, **kwargs):
        super(Door, self).__init__(*args, **kwargs)
        self._state = c.CLOSED_DOOR
        self.has_area = has_area
        self.auto_close_interval = auto_close_interval
        self.time_to_close = -1
        neighbor_pos = list(itertools.product([-1, 1, 0], repeat=2))[:-1]
        neighbor_tiles = [context.by_pos(tuple([sum(x) for x in zip(self.pos, diff)])) for diff in neighbor_pos]
        neighbor_pos = [x.pos for x in neighbor_tiles if x]
        self.connectivity = h.points_to_graph(neighbor_pos)
        self.connectivity_subgroups = list(nx.algorithms.components.connected_components(self.connectivity))
        for idx, group in enumerate(self.connectivity_subgroups):
            for tile_pos in group:
                self.connectivity.add_edge(tile_pos, idx)
        if not closed_on_init:
            self._open()

    def summarize_state(self, **kwargs):
        state_dict = super().summarize_state(**kwargs)
        state_dict.update(state=str(self.str_state), time_to_close=int(self.time_to_close))
        return state_dict

    @property
    def is_closed(self):
        return self._state == c.CLOSED_DOOR

    @property
    def is_open(self):
        return self._state == c.OPEN_DOOR

    @property
    def status(self):
        return self._state

    def use(self):
        if self._state == c.OPEN_DOOR:
            self._close()
        else:
            self._open()

    def tick(self):
        if self.is_open and len(self.tile) == 1 and self.time_to_close:
            self.time_to_close -= 1
        elif self.is_open and not self.time_to_close and len(self.tile) == 1:
            self.use()

    def _open(self):
        self.connectivity.add_edges_from([(self.pos, x) for x in range(len(self.connectivity_subgroups))])
        self._state = c.OPEN_DOOR
        self._register.notify_change_to_value(self)
        self.time_to_close = self.auto_close_interval

    def _close(self):
        self.connectivity.remove_node(self.pos)
        self._state = c.CLOSED_DOOR
        self._register.notify_change_to_value(self)

    def is_linked(self, old_pos, new_pos):
        try:
            _ = nx.shortest_path(self.connectivity, old_pos, new_pos)
            return True
        except nx.exception.NetworkXNoPath:
            return False


class Agent(MoveableEntity):

    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        self.clear_temp_state()

    # noinspection PyAttributeOutsideInit
    def clear_temp_state(self):
        # for attr in self.__dict__:
        #   if attr.startswith('temp'):
        self.temp_collisions = []
        self.temp_valid = None
        self.temp_action = None
        self.temp_light_map = None

    def summarize_state(self, **kwargs):
        state_dict = super().summarize_state(**kwargs)
        state_dict.update(valid=bool(self.temp_valid), action=str(self.temp_action))
        return state_dict
