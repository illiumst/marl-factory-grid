import numpy as np

from marl_factory_grid.environment.entity.object import Object


##########################################################################
# ####################### Objects and Entities ########################## #
##########################################################################


class PlaceHolder(Object):

    def __init__(self, *args, fill_value=0, **kwargs):
        """
        A placeholder object that can be used as an observation during training. It is designed to be later replaced
        with a meaningful observation that wasn't initially present in the training run.

        :param fill_value: The default value to fill the placeholder observation (Default: 0)
        :type fill_value: Any
        """
        super().__init__(*args, **kwargs)
        self._fill_value = fill_value

    @property
    def var_can_collide(self):
        """
        Indicates whether this placeholder object can collide with other entities. Always returns False.

        :return: False
        :rtype: bool
        """
        return False

    @property
    def encoding(self):
        """
        Get the fill value representing the placeholder observation.

        :return: The fill value
        :rtype: Any
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
        Get the encoded representation of the global position based on whether normalization is enabled.

        :return: The encoded representation of the global position
        :rtype: tuple[float, float] or tuple[int, int]
        """
        if self._normalized:
            return tuple(np.divide(self._bound_entity.pos, self._shape))
        else:
            return self.bound_entity.pos

    def __init__(self, agent, level_shape, *args, normalized: bool = True, **kwargs):
        """
        A utility class representing the global position of an entity in the environment.

        :param agent: The agent entity to which the global position is bound.
        :type agent: marl_factory_grid.environment.entity.agent.Agent
        :param level_shape: The shape of the environment level.
        :type level_shape: tuple[int, int]
        :param normalized: Indicates whether the global position should be normalized (Default: True)
        :type normalized: bool
        """
        super(GlobalPosition, self).__init__(*args, **kwargs)
        self.bind_to(agent)
        self._normalized = normalized
        self._shape = level_shape
