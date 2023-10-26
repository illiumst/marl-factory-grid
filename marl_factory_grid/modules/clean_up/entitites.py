from numpy import random

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.modules.clean_up import constants as d


class DirtPile(Entity):

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_move(self):
        return False

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_has_position(self):
        return True

    @property
    def amount(self):
        return self._amount

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differntly
        return self._amount

    def __init__(self, *args, amount=2, max_local_amount=5, **kwargs):
        super(DirtPile, self).__init__(*args, **kwargs)
        self._amount = amount
        self.max_local_amount = max_local_amount

    def set_new_amount(self, amount):
        self._amount = min(amount, self.max_local_amount)

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(amount=float(self.amount))
        return state_dict

    def render(self):
        return RenderEntity(d.DIRT, self.pos, min(0.15 + self.amount, 1.5), 'scale')
