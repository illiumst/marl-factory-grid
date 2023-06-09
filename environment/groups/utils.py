import numbers
from typing import List, Union, Dict

import numpy as np

from environment.groups.env_objects import EnvObjects
from environment.groups.objects import Objects
from environment.groups.mixins import HasBoundedMixin, PositionMixin
from environment.entity.util import PlaceHolder, GlobalPosition
from environment.utils import helpers as h
from environment import constants as c


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


class GlobalPositions(HasBoundedMixin, EnvObjects):

    _entity = GlobalPosition
    is_blocking_light = False,
    can_collide = False

    def __init__(self, *args, **kwargs):
        super(GlobalPositions, self).__init__(*args, **kwargs)


class Zones(Objects):

    @property
    def accounting_zones(self):
        return [self[idx] for idx, name in self.items() if name != c.DANGER_ZONE]

    def __init__(self, parsed_level):
        raise NotImplementedError('This needs a Rework')
        super(Zones, self).__init__()
        slices = list()
        self._accounting_zones = list()
        self._danger_zones = list()
        for symbol in np.unique(parsed_level):
            if symbol == c.VALUE_OCCUPIED_CELL:
                continue
            elif symbol == c.DANGER_ZONE:
                self + symbol
                slices.append(h.one_hot_level(parsed_level, symbol))
                self._danger_zones.append(symbol)
            else:
                self + symbol
                slices.append(h.one_hot_level(parsed_level, symbol))
                self._accounting_zones.append(symbol)

        self._zone_slices = np.stack(slices)

    def __getitem__(self, item):
        return self._zone_slices[item]

    def add_items(self, other: Union[str, List[str]]):
        raise AttributeError('You are not allowed to add additional Zones in runtime.')
