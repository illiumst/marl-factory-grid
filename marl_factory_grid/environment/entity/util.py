import numpy as np

from marl_factory_grid.environment.entity.object import Object


##########################################################################
# ####################### Objects and Entitys ########################## #
##########################################################################


class PlaceHolder(Object):

    def __init__(self, *args, fill_value=0, **kwargs):
        """
        TODO


        :return:
        """
        super().__init__(*args, **kwargs)
        self._fill_value = fill_value

    @property
    def var_can_collide(self):
        """
        TODO


        :return:
        """
        return False

    @property
    def encoding(self):
        """
        TODO


        :return:
        """
        return self._fill_value

    @property
    def name(self):
        return self.__class__.__name__


class GlobalPosition(Object):

    @property
    def obs_tag(self):
        return self.name

    @property
    def encoding(self):
        """
        TODO


        :return:
        """
        if self._normalized:
            return tuple(np.divide(self._bound_entity.pos, self._shape))
        else:
            return self.bound_entity.pos

    def __init__(self, agent, level_shape, *args, normalized: bool = True, **kwargs):
        """
        TODO


        :return:
        """
        super(GlobalPosition, self).__init__(*args, **kwargs)
        self.bind_to(agent)
        self._normalized = normalized
        self._shape = level_shape
