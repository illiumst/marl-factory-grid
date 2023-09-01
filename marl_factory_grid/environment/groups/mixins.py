from typing import List

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.entity.wall_floor import Floor


class PositionMixin:

    _entity = Entity
    var_is_blocking_light: bool = True
    var_can_collide: bool = True
    var_has_position: bool = True

    def spawn(self, tiles: List[Floor]):
        self.add_items([self._entity(tile) for tile in tiles])

    def render(self):
        return [y for y in [x.render() for x in self] if y is not None]

    @classmethod
    def from_tiles(cls, tiles, *args, entity_kwargs=None, **kwargs):
        collection = cls(*args, **kwargs)
        entities = [cls._entity(tile, str_ident=i,
                                **entity_kwargs if entity_kwargs is not None else {})
                    for i, tile in enumerate(tiles)]
        collection.add_items(entities)
        return collection

    @classmethod
    def from_coordinates(cls, positions: [(int, int)], tiles, *args, entity_kwargs=None, **kwargs, ):
        return cls.from_tiles([tiles.by_pos(position) for position in positions], tiles.size, *args,
                              entity_kwargs=entity_kwargs,
                              **kwargs)

    @property
    def tiles(self):
        return [entity.tile for entity in self]

    def __delitem__(self, name):
        idx, obj = next((i, obj) for i, obj in enumerate(self) if obj.name == name)
        obj.tile.leave(obj)
        super().__delitem__(name)

    def by_pos(self, pos: (int, int)):
        pos = tuple(pos)
        try:
            return self.pos_dict[pos]
            # return next(e for e in self if e.pos == pos)
        except StopIteration:
            pass
        except ValueError:
            print()

    @property
    def positions(self):
        return [e.pos for e in self]

    def notify_del_entity(self, entity: Entity):
        try:
            self.pos_dict[entity.pos].remove(entity)
        except (ValueError, AttributeError):
            pass


# noinspection PyUnresolvedReferences,PyTypeChecker
class IsBoundMixin:

    @property
    def name(self):
        return f'{self.__class__.__name__}({self._bound_entity.name})'

    def __repr__(self):
        return f'{self.__class__.__name__}#{self._bound_entity.name}({self._data})'

    def bind(self, entity):
        # noinspection PyAttributeOutsideInit
        self._bound_entity = entity
        return c.VALID

    def belongs_to_entity(self, entity):
        return self._bound_entity == entity


# noinspection PyUnresolvedReferences,PyTypeChecker
class HasBoundMixin:

    @property
    def obs_pairs(self):
        return [(x.name, x) for x in self]

    def by_entity(self, entity):
        try:
            return next((x for x in self if x.belongs_to_entity(entity)))
        except StopIteration:
            return None

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, x in enumerate(self) if x.belongs_to_entity(entity)))
        except StopIteration:
            return None
