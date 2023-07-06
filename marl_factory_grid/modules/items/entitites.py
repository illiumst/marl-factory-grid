from collections import deque

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.modules.items import constants as i


class Item(Entity):

    var_can_collide = False

    def render(self):
        return RenderEntity(i.ITEM, self.tile.pos) if self.pos != c.VALUE_NO_POS else None

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

    def render(self):
        return RenderEntity(i.DROP_OFF, self.tile.pos)

    @property
    def encoding(self):
        return i.SYMBOL_DROP_OFF

    def __init__(self, *args, storage_size_until_full: int = 5, auto_item_despawn_interval: int = 5, **kwargs):
        super(DropOffLocation, self).__init__(*args, **kwargs)
        self.auto_item_despawn_interval = auto_item_despawn_interval
        self.storage = deque(maxlen=storage_size_until_full or None)

    def place_item(self, item: Item):
        if self.is_full:
            raise RuntimeWarning("There is currently no way to clear the storage or make it unfull.")
            return bc.NOT_VALID
        else:
            self.storage.append(item)
            item.set_auto_despawn(self.auto_item_despawn_interval)
            return c.VALID

    @property
    def is_full(self):
        return False if not self.storage.maxlen else self.storage.maxlen == len(self.storage)
