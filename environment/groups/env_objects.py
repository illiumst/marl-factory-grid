from environment.groups.objects import Objects
from environment.entity.object import EnvObject


class EnvObjects(Objects):

    _entity = EnvObject
    is_blocking_light: bool = False
    can_collide: bool = False
    has_position: bool = False
    can_move: bool = False

    @property
    def encodings(self):
        return [x.encoding for x in self]

    def __init__(self, size, *args, **kwargs):
        super(EnvObjects, self).__init__(*args, **kwargs)
        self.size = size

    def add_item(self, item: EnvObject):
        assert self.has_position or (len(self) <= self.size)
        super(EnvObjects, self).add_item(item)
        return self

    def summarize_states(self):
        return [entity.summarize_state() for entity in self.values()]

    def delete_env_object(self, env_object: EnvObject):
        del self[env_object.name]

    def delete_env_object_by_name(self, name):
        del self[name]
