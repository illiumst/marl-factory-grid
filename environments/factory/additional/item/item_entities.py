from collections import deque

from environments import helpers as h
from environments.factory.additional.item.item_util import Constants
from environments.factory.base.objects import Entity


class Item(Entity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_despawn = -1

    @property
    def auto_despawn(self):
        return self._auto_despawn

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differently
        return 1

    def set_auto_despawn(self, auto_despawn):
        self._auto_despawn = auto_despawn

    def set_tile_to(self, no_pos_tile):
        self._tile = no_pos_tile

    def summarize_state(self) -> dict:
        super_summarization = super(Item, self).summarize_state()
        super_summarization.update(dict(auto_despawn=self.auto_despawn))
        return super_summarization


class DropOffLocation(Entity):

    @property
    def encoding(self):
        return Constants.ITEM_DROP_OFF

    def __init__(self, *args, storage_size_until_full: int = 5, auto_item_despawn_interval: int = 5, **kwargs):
        super(DropOffLocation, self).__init__(*args, **kwargs)
        self.auto_item_despawn_interval = auto_item_despawn_interval
        self.storage = deque(maxlen=storage_size_until_full or None)

    def place_item(self, item: Item):
        if self.is_full:
            raise RuntimeWarning("There is currently no way to clear the storage or make it unfull.")
            return c.NOT_VALID
        else:
            self.storage.append(item)
            item.set_auto_despawn(self.auto_item_despawn_interval)
            return Constants.VALID

    @property
    def is_full(self):
        return False if not self.storage.maxlen else self.storage.maxlen == len(self.storage)
