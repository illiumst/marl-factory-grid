

# noinspection PyAttributeOutsideInit
class BoundEntityMixin:

    @property
    def bound_entity(self):
        return self._bound_entity

    @property
    def name(self):
        return f'{self.__class__.__name__}({self.bound_entity.name})'

    def belongs_to_entity(self, entity):
        return entity == self.bound_entity

    def bind_to(self, entity):
        self._bound_entity = entity
