from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.environment.entity.object import EnvObject


class EnvObjects(Objects):

    _entity = EnvObject
    var_is_blocking_light: bool = False
    var_can_collide: bool = False
    var_has_position: bool = False
    var_can_move: bool = False

    @property
    def encodings(self):
        return [x.encoding for x in self]

    def __init__(self, size, *args, **kwargs):
        super(EnvObjects, self).__init__(*args, **kwargs)
        self.size = size

    def add_item(self, item: EnvObject):
        assert self.var_has_position or (len(self) <= self.size)
        super(EnvObjects, self).add_item(item)
        return self

    def delete_env_object(self, env_object: EnvObject):
        del self[env_object.name]

    def delete_env_object_by_name(self, name):
        del self[name]
