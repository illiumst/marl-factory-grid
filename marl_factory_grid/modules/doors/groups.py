from typing import Union

from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin
from marl_factory_grid.modules.doors import constants as d
from marl_factory_grid.modules.doors.entitites import Door


class Doors(PositionMixin, EnvObjects):

    symbol = d.SYMBOL_DOOR
    _entity = Door

    def __init__(self, *args, **kwargs):
        super(Doors, self).__init__(*args, can_collide=True, **kwargs)

    def tick_doors(self):
        result_dict = dict()
        for door in self:
            did_tick = door.tick()
            result_dict.update({door.name: did_tick})
        return result_dict
