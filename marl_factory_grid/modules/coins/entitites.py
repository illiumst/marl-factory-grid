from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.modules.coins import constants as d


class CoinPile(Entity):

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
        Represents a pile of coins at a specific position in the environment that agents can interact with. Agents can
        clean the dirt pile or, depending on activated rules, interact with it in different ways.

        :param amount: The amount of coins in the pile.
        :type amount: float

        :param max_local_amount: The maximum amount of dirt allowed in a single pile at one position.
        :type max_local_amount: float
        """
        super(CoinPile, self).__init__(*args, **kwargs)
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
        return RenderEntity(d.COIN, self.pos, min(0 + self.amount, 1.5), 'scale')
