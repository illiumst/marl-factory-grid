from collections import defaultdict
from typing import Union, List

import numpy as np

from environments import helpers as h
from environments.helpers import Constants as c

##########################################################################
# ##################### Base Object Building Blocks ######################### #
##########################################################################


# TODO: Missing Documentation
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
        if self._str_ident is not None:
            return self._str_ident
        else:
            return self._name

    def __init__(self, str_ident: Union[str, None] = None, **kwargs):

        self._str_ident = str_ident

        if self._str_ident is not None:
            self._name = f'{self.__class__.__name__}[{self._str_ident}]'
        elif self._str_ident is None:
            self._name = f'{self.__class__.__name__}#{Object._u_idx[self.__class__.__name__]}'
            Object._u_idx[self.__class__.__name__] += 1
        else:
            raise ValueError('Please use either of the idents.')

        if kwargs:
            print(f'Following kwargs were passed, but ignored: {kwargs}')

    def __repr__(self):
        return f'{self.name}'

    def __eq__(self, other) -> bool:
        return other == self.identifier
# Base


# TODO: Missing Documentation
class EnvObject(Object):

    """Objects that hold Information that are observable, but have no position on the env grid. Inventories etc..."""

    _u_idx = defaultdict(lambda: 0)

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        return c.OCCUPIED_CELL

    def __init__(self, collection, **kwargs):
        super(EnvObject, self).__init__(**kwargs)
        self._collection = collection

    def change_parent_collection(self, other_collection):
        other_collection.add_item(self)
        self._collection.delete_env_object(self)
        self._collection = other_collection
        return self._collection == other_collection
# With Rendering


# TODO: Missing Documentation
class Entity(EnvObject):
    """Full Env Entity that lives on the env Grid. Doors, Items, DirtPile etc..."""

    @property
    def is_blocking(self):
        return False

    @property
    def can_collide(self):
        return False

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

    def __init__(self, tile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tile = tile
        tile.enter(self)

    def summarize_state(self) -> dict:
        return dict(name=str(self.name), x=int(self.x), y=int(self.y),
                    tile=str(self.tile.name), can_collide=bool(self.can_collide))

    def __repr__(self):
        return super(Entity, self).__repr__() + f'(@{self.pos})'


# TODO: Missing Documentation
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
        super().__init__(*args, **kwargs)
        self._last_tile = None

    def move(self, next_tile):
        curr_tile = self.tile
        if curr_tile != next_tile:
            next_tile.enter(self)
            curr_tile.leave(self)
            self._tile = next_tile
            self._last_tile = curr_tile
            self._collection.notify_change_to_value(self)
            return c.VALID
        else:
            return c.NOT_VALID
# Can Move


# TODO: Missing Documentation
class BoundingMixin(Object):

    @property
    def bound_entity(self):
        return self._bound_entity

    def __init__(self,entity_to_be_bound, *args, **kwargs):
        super(BoundingMixin, self).__init__(*args, **kwargs)
        assert entity_to_be_bound is not None
        self._bound_entity = entity_to_be_bound

    @property
    def name(self):
        return f'{super(BoundingMixin, self).name}({self._bound_entity.name})'

    def belongs_to_entity(self, entity):
        return entity == self.bound_entity


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


class GlobalPosition(BoundingMixin, EnvObject):

    @property
    def encoding(self):
        if self._normalized:
            return tuple(np.divide(self._bound_entity.pos, self._level_shape))
        else:
            return self.bound_entity.pos

    def __init__(self, level_shape: (int, int), *args, normalized: bool = True, **kwargs):
        super(GlobalPosition, self).__init__(*args, **kwargs)
        self._level_shape = level_shape
        self._normalized = normalized


class Floor(EnvObject):

    @property
    def neighboring_floor_pos(self):
        return [x.pos for x in self.neighboring_floor]

    @property
    def neighboring_floor(self):
        if self._neighboring_floor:
            pass
        else:
            self._neighboring_floor = [x for x in [self._collection.by_pos(np.add(self.pos, pos))
                                                   for pos in h.POS_MASK.reshape(-1, 2)
                                                   if not np.all(pos == [0, 0])]
                                       if x]
        return self._neighboring_floor

    @property
    def encoding(self):
        return c.FREE_CELL

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
        super(Floor, self).__init__(*args, **kwargs)
        self._guests = dict()
        self._pos = tuple(pos)
        self._neighboring_floor: List[Floor] = list()

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


class Wall(Floor):

    @property
    def can_collide(self):
        return True

    @property
    def encoding(self):
        return c.OCCUPIED_CELL

    pass


class Agent(MoveableEntity):

    @property
    def can_collide(self):
        return True

    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        self.clear_temp_state()

    # noinspection PyAttributeOutsideInit
    def clear_temp_state(self):
        # for attr in cls.__dict__:
        #   if attr.startswith('temp'):
        self.step_result = None

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(valid=bool(self.step_result['action_valid']), action=str(self.step_result['action_name']))
        return state_dict
