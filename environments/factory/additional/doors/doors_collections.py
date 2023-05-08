from typing import Union

from environments.factory.additional.doors.doors_entities import Door
from environments.factory.base.registers import EntityCollection

from environments.factory.additional.doors.doors_util import Constants as c


class Doors(EntityCollection):

    def __init__(self, *args, indicate_area=False, **kwargs):
        self.indicate_area = indicate_area
        self._area_marked = False
        super(Doors, self).__init__(*args, is_blocking_light=True, can_collide=True, **kwargs)

    _accepted_objects = Door

    def get_near_position(self, position: (int, int)) -> Union[None, Door]:
        try:
            return next(door for door in self if position in door.tile.neighboring_floor_pos)
        except StopIteration:
            return None

    def tick_doors(self):
        for door in self:
            door.tick()

    def as_array(self):
        if not self._area_marked and self.indicate_area:
            for door in self:
                for tile in door.tile.neighboring_floor:
                    if self._individual_slices:
                        pass
                    else:
                        pos = (0, *tile.pos)
                    self._lazy_eval_transforms.append((pos, c.ACCESS_DOOR_CELL))
            self._area_marked = True
        return super(Doors, self).as_array()
