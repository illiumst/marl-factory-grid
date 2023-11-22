from collections import deque

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.modules.items import constants as i


class Item(Entity):

    @property
    def encoding(self):
        return 1

    def __init__(self, *args, **kwargs):
        """
        An item that can be picked up or dropped by agents. If picked up, it enters the agents inventory.
        """
        super().__init__(*args, **kwargs)

    def render(self):
        return RenderEntity(i.ITEM, self.pos) if self.pos != c.VALUE_NO_POS else None


class DropOffLocation(Entity):
    @property
    def encoding(self):
        return i.SYMBOL_DROP_OFF

    @property
    def is_full(self) -> bool:
        """
        Checks whether the drop-off location is full or whether another item can be dropped here.
        """
        return False if not self.storage.maxlen else self.storage.maxlen == len(self.storage)

    def __init__(self, *args, storage_size_until_full=5, **kwargs):
        """
        Represents a drop-off location in the environment that agents aim to drop items at.

        :param storage_size_until_full: The number of items that can be dropped here until it is considered full.
        :type storage_size_until_full: int
        """
        super(DropOffLocation, self).__init__(*args, **kwargs)
        self.storage = deque(maxlen=storage_size_until_full or None)

    def place_item(self, item: Item) -> bool:
        """
        If the storage of the drop-off location is not full, the item is placed. Otherwise, a RuntimeWarning is raised.
        """
        if self.is_full:
            raise RuntimeWarning("There is currently no way to clear the storage or make it unfull.")
            return c.NOT_VALID
        else:
            self.storage.append(item)
            return c.VALID

    def render(self):
        return RenderEntity(i.DROP_OFF, self.pos)
