from typing import List, Union

import numpy as np

from marl_factory_grid.environment.entity.util import GlobalPosition
from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin, HasBoundMixin
from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.modules.zones import Zone
from marl_factory_grid.utils import helpers as h
from marl_factory_grid.environment import constants as c


class Combined(PositionMixin, EnvObjects):

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


class GlobalPositions(HasBoundMixin, EnvObjects):

    _entity = GlobalPosition
    is_blocking_light = False,
    can_collide = False

    def __init__(self, *args, **kwargs):
        super(GlobalPositions, self).__init__(*args, **kwargs)
