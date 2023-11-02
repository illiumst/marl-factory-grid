from typing import List, Union

from marl_factory_grid.environment.entity.util import GlobalPosition
from marl_factory_grid.environment.groups.collection import Collection


class Combined(Collection):

    @property
    def var_has_position(self):
        return True

    @property
    def name(self):
        return f'{super().name}({self._ident or self._names})'

    @property
    def names(self):
        return self._names

    def __init__(self, names: List[str], *args, identifier: Union[None, str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ident = identifier
        self._names = names or list()

    @property
    def obs_tag(self):
        return self.name

    @property
    def obs_pairs(self):
        return [(name, None) for name in self.names]


class GlobalPositions(Collection):

    _entity = GlobalPosition

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_be_bound(self):
        return True

    def __init__(self, *args, **kwargs):
        super(GlobalPositions, self).__init__(*args, **kwargs)
