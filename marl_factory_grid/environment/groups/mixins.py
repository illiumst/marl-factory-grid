from marl_factory_grid.environment import constants as c


# noinspection PyUnresolvedReferences,PyTypeChecker
class IsBoundMixin:

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
        except (StopIteration, AttributeError):
            return None
