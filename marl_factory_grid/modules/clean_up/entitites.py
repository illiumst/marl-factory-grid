from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.modules.clean_up import constants as d


class DirtPile(Entity):

    @property
    def amount(self):
        """
        Internal Usage
        """
        return self._amount

    @property
    def encoding(self):
        return self._amount

    def __init__(self, *args, amount=2, max_local_amount=5, **kwargs):
        """
        Represents a pile of dirt at a specific position in the environment.

        :param amount: The amount of dirt in the pile.
        :type amount: float

        :param max_local_amount: The maximum amount of dirt allowed in a single pile at one position.
        :type max_local_amount: float
        """
        super(DirtPile, self).__init__(*args, **kwargs)
        self._amount = amount
        self.max_local_amount = max_local_amount

    def set_new_amount(self, amount):
        """
        Internal Usage
        """
        self._amount = min(amount, self.max_local_amount)

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(amount=float(self.amount))
        return state_dict

    def render(self):
        return RenderEntity(d.DIRT, self.pos, min(0.15 + self.amount, 1.5), 'scale')
