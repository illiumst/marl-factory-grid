import numpy as np

from marl_factory_grid.environment.entity.object import Object


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
        return self.__class__.__name__


class GlobalPosition(Object):

    @property
    def encoding(self):
        if self._normalized:
            return tuple(np.divide(self._bound_entity.pos, self._shape))
        else:
            return self.bound_entity.pos

    def __init__(self, agent, level_shape, *args, normalized: bool = True, **kwargs):
        super(GlobalPosition, self).__init__(*args, **kwargs)
        self.bind_to(agent)
        self._normalized = normalized
        self._shape = level_shape
