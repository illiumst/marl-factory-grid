from marl_factory_grid.environment.groups.objects import _Objects
from marl_factory_grid.environment.entity.object import EnvObject


class Collection(_Objects):
    _entity = EnvObject

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_move(self):
        return False

    @property
    def var_has_position(self):
        return False  # alles was posmixin hat true
    @property
    def var_has_bound(self):
        return False  # batteries, globalpos, inventories true

    @property   # beide bounds hier? inventory can be bound
    def var_can_be_bound(self):
        return False

    @property
    def encodings(self):
        return [x.encoding for x in self]

    def __init__(self, size, *args, **kwargs):
        super(Collection, self).__init__(*args, **kwargs)
        self.size = size

    def add_item(self, item: EnvObject):
        assert self.var_has_position or (len(self) <= self.size)
        super(Collection, self).add_item(item)
        return self

    def delete_env_object(self, env_object: EnvObject):
        del self[env_object.name]

    def delete_env_object_by_name(self, name):
        del self[name]

    @property
    def obs_pairs(self):
        return [(x.name, x) for x in self]

    def by_entity(self, entity):
        try:
            return next((x for x in self if x.belongs_to_entity(entity)))
        except (StopIteration, AttributeError):
            return None

    def idx_by_entity(self, entity):
        try:
            return next((idx for idx, x in enumerate(self) if x.belongs_to_entity(entity)))
        except (StopIteration, AttributeError):
            return None
