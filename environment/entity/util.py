import math

import numpy as np

from environment.entity.mixin import BoundEntityMixin
from environment.entity.object import Object, EnvObject


##########################################################################
# ####################### Objects and Entitys ########################## #
##########################################################################


class PlaceHolder(Object):

    def __init__(self, *args, fill_value=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._fill_value = fill_value

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        return self._fill_value

    @property
    def name(self):
        return "PlaceHolder"


class GlobalPosition(BoundEntityMixin, EnvObject):

    @property
    def encoding(self):
        if self._normalized:
            return tuple(np.divide(self._bound_entity.pos, self._level_shape))
        else:
            return self.bound_entity.pos

    def __init__(self, *args, normalized: bool = True, **kwargs):
        super(GlobalPosition, self).__init__(*args, **kwargs)
        self._level_shape = math.sqrt(self.size)
        self._normalized = normalized
