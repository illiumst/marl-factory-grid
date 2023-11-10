from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.doors import constants as d
from marl_factory_grid.modules.doors.entitites import Door


class Doors(Collection):

    symbol = d.SYMBOL_DOOR
    _entity = Door

    @property
    def var_has_position(self):
        return True

    def __init__(self, *args, **kwargs):
        super(Doors, self).__init__(*args, can_collide=True, **kwargs)

    def tick_doors(self, state):
        results = list()
        for door in self:
            tick_result = door.tick(state)
            if tick_result is not None:
                results.append(tick_result)
        # TODO: Should return a Result object, not a random dict.
        return results

    def reset(self):
        for door in self:
            door.reset()
