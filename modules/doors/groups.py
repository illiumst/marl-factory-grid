from typing import Union

from environment.groups.env_objects import EnvObjects
from environment.groups.mixins import PositionMixin
from modules.doors import constants as d
from modules.doors.entitites import Door


class Doors(PositionMixin, EnvObjects):

    symbol = d.SYMBOL_DOOR
    _entity = Door

    def __init__(self, *args, **kwargs):
        super(Doors, self).__init__(*args, can_collide=True, **kwargs)

    def get_near_position(self, position: (int, int)) -> Union[None, Door]:
        try:
            return next(door for door in self if position in door.tile.neighboring_floor_pos)
        except StopIteration:
            return None

    def tick_doors(self):
        result_dict = dict()
        for door in self:
            did_tick = door.tick()
            result_dict.update({door.name: did_tick})
        return result_dict
