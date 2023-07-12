from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.modules.zones import Zone


class Zones(Objects):

    symbol = None
    _entity = Zone
    var_can_move = False

    def __init__(self, *args, **kwargs):
        super(Zones, self).__init__(*args, can_collide=True, **kwargs)

    def by_pos(self, pos):
        return self.pos_dict[pos]

    def notify_add_entity(self, entity: Zone):
        self.pos_dict.update({key: [entity] for key in entity.positions})
        return True

    def notify_del_entity(self, entity: Zone):
        for pos in entity.positions:
            self.pos_dict[pos].remove(entity)
        return True
