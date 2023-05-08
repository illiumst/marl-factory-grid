from environments.factory.additional.dirt.dirt_entity import DirtPile
from environments.factory.additional.dirt.dirt_util import DirtProperties
from environments.factory.base.objects import Floor
from environments.factory.base.registers import EntityCollection
from environments.factory.additional.dirt.dirt_util import Constants as c


class DirtPiles(EntityCollection):

    _accepted_objects = DirtPile

    @property
    def amount(self):
        return sum([dirt.amount for dirt in self])

    @property
    def dirt_properties(self):
        return self._dirt_properties

    def __init__(self, dirt_properties, *args):
        super(DirtPiles, self).__init__(*args)
        self._dirt_properties: DirtProperties = dirt_properties

    def spawn_dirt(self, then_dirty_tiles) -> bool:
        if isinstance(then_dirty_tiles, Floor):
            then_dirty_tiles = [then_dirty_tiles]
        for tile in then_dirty_tiles:
            if not self.amount > self.dirt_properties.max_global_amount:
                dirt = self.by_pos(tile.pos)
                if dirt is None:
                    if len(tile.guests) > 1:
                        return c.NOT_VALID
                    dirt = DirtPile(tile, self, amount=self.dirt_properties.max_spawn_amount)
                    self.add_item(dirt)
                else:
                    new_value = dirt.amount + self.dirt_properties.max_spawn_amount
                    dirt.set_new_amount(min(new_value, self.dirt_properties.max_local_amount))
            else:
                return c.NOT_VALID
        return c.VALID

    def __repr__(self):
        s = super(DirtPiles, self).__repr__()
        return f'{s[:-1]}, {self.amount})'
