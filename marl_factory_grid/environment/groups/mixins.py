from marl_factory_grid.environment import constants as c

"""
Mixins are a way to modularly extend the functionality of classes in object-oriented programming without using 
inheritance in the traditional sense. They provide a means to include a set of methods and properties in a class that 
can be reused across different class hierarchies.
"""


# noinspection PyUnresolvedReferences,PyTypeChecker
class IsBoundMixin:
    """
    This mixin is designed to be used in classes that represent objects which can be bound to another entity.
    """

    def __repr__(self):
        return f'{self.__class__.__name__}#{self._bound_entity.name}({self._data})'

    def bind(self, entity):
        """
        Binds the current object to another entity.

        :param entity: the entity to be bound
        """
        # noinspection PyAttributeOutsideInit
        self._bound_entity = entity
        return c.VALID

    def belongs_to_entity(self, entity):
        """
        Checks if the given entity is the bound entity.

        :return: True if the given entity is the bound entity, false otherwise.
        """
        return self._bound_entity == entity


# noinspection PyUnresolvedReferences,PyTypeChecker
class HasBoundMixin:
    """
    This mixin is intended for classes that contain a collection of objects and need functionality to interact with
    those objects.
    """

    @property
    def obs_pairs(self):
        """
        Returns a list of pairs containing the names and corresponding objects within the collection.
        """
        return [(x.name, x) for x in self]

    def by_entity(self, entity):
        """
        Retrieves an object from the collection based on its belonging to a specific entity.
        """
        try:
            return next((x for x in self if x.belongs_to_entity(entity)))
        except (StopIteration, AttributeError):
            return None
