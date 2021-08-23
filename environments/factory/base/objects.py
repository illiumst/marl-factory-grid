import networkx as nx
import numpy as np
from environments.helpers import Constants as c
import itertools


def sub(p, q):
    return p - q


class Object:

    def __bool__(self):
        return True

    @property
    def name(self):
        return self._name

    def __init__(self, name, name_is_identifier=False, **kwargs):
        name = name.name if hasattr(name, 'name') else name
        self._name = f'{self.__class__.__name__}#{name}' if name_is_identifier else name
        if kwargs:
            print(f'Following kwargs were passed, but ignored: {kwargs}')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'


class Action(Object):

    def __init__(self, *args):
        super(Action, self).__init__(*args)


class Slice(Object):

    @property
    def is_observable(self):
        return self._is_observable

    @property
    def shape(self):
        return self.slice.shape

    @property
    def occupied_tiles(self):
        return np.argwhere(self.slice == c.OCCUPIED_CELL.value)

    @property
    def free_tiles(self):
        return np.argwhere(self.slice == c.FREE_CELL.value)

    def __init__(self, identifier, arrayslice, is_blocking_light=False, can_be_shadowed=True, is_observable=True):
        super(Slice, self).__init__(identifier)
        self.slice = arrayslice
        self.is_blocking_light = is_blocking_light
        self.can_be_shadowed = can_be_shadowed
        self._is_observable = is_observable

    def set_slice(self, new_slice: np.ndarray):
        assert self.slice.shape == new_slice.shape
        self.slice = new_slice


class Wall(Object):
    pass


class Tile(Object):

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

    def __init__(self, i, pos, **kwargs):
        super(Tile, self).__init__(i, **kwargs)
        self._guests = dict()
        self._pos = tuple(pos)

    def __len__(self):
        return len(self._guests)

    def is_empty(self):
        return not len(self._guests)

    def is_occupied(self):
        return len(self._guests)

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


class Entity(Object):

    @property
    def can_collide(self):
        return True

    @property
    def encoding(self):
        return 1

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

    def __init__(self, identifier, tile: Tile, **kwargs):
        super(Entity, self).__init__(identifier, **kwargs)
        self._tile = tile
        tile.enter(self)

    def summarize_state(self):
        return self.__dict__.copy()


class MoveableEntity(Entity):

    @property
    def last_tile(self):
        return self._last_tile

    @property
    def last_pos(self):
        if self._last_tile:
            return self._last_tile.pos
        else:
            return c.NO_POS

    @property
    def direction_of_view(self):
        last_x, last_y = self.last_pos
        curr_x, curr_y = self.pos
        return last_x-curr_x, last_y-curr_y

    def __init__(self, *args, **kwargs):
        super(MoveableEntity, self).__init__(*args, **kwargs)
        self._last_tile = None

    def move(self, next_tile):
        curr_tile = self.tile
        if curr_tile != next_tile:
            next_tile.enter(self)
            curr_tile.leave(self)
            self._tile = next_tile
            self._last_tile = curr_tile
            return True
        else:
            return False


class Door(Entity):

    @property
    def can_collide(self):
        if self.has_area:
            return False if self.is_open else True
        else:
            return False

    @property
    def encoding(self):
        return 1 if self.is_closed else -1

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
        possible_connections = itertools.combinations(neighbor_pos, 2)
        self.connectivity = nx.Graph()
        for a, b in possible_connections:
            if not max(abs(np.subtract(a, b))) > 1:
                self.connectivity.add_edge(a, b)
        self.connectivity_subgroups = list(nx.algorithms.components.connected_components(self.connectivity))
        for idx, group in enumerate(self.connectivity_subgroups):
            for tile_pos in group:
                self.connectivity.add_edge(tile_pos, idx)
        if not closed_on_init:
            self._open()

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
        self.time_to_close = self.auto_close_interval

    def _close(self):
        self.connectivity.remove_node(self.pos)
        self._state = c.CLOSED_DOOR

    def is_linked(self, old_pos, new_pos):
        try:
            _ = nx.shortest_path(self.connectivity, old_pos, new_pos)
            return True
        except nx.exception.NetworkXNoPath:
            return False


class Agent(MoveableEntity):

    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        self.clear_temp_sate()

    # noinspection PyAttributeOutsideInit
    def clear_temp_sate(self):
        # for attr in self.__dict__:
        #   if attr.startswith('temp'):
        self.temp_collisions = []
        self.temp_valid = None
        self.temp_action = None
        self.temp_light_map = None
