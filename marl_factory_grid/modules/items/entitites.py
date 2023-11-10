from collections import deque

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.modules.items import constants as i


class Item(Entity):

    def render(self):
        return RenderEntity(i.ITEM, self.pos) if self.pos != c.VALUE_NO_POS else None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differently
        return 1


class DropOffLocation(Entity):

    def render(self):
        return RenderEntity(i.DROP_OFF, self.pos)

    @property
    def encoding(self):
        return i.SYMBOL_DROP_OFF

    def __init__(self, *args, storage_size_until_full: int = 5, **kwargs):
        super(DropOffLocation, self).__init__(*args, **kwargs)
        self.storage = deque(maxlen=storage_size_until_full or None)

    def place_item(self, item: Item):
        if self.is_full:
            raise RuntimeWarning("There is currently no way to clear the storage or make it unfull.")
            return bc.NOT_VALID
        else:
            self.storage.append(item)
            return c.VALID

    @property
    def is_full(self):
        return False if not self.storage.maxlen else self.storage.maxlen == len(self.storage)
