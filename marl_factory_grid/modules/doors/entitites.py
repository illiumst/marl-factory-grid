from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.doors import constants as d


class DoorIndicator(Entity):

    @property
    def encoding(self):
        return d.VALUE_ACCESS_INDICATOR

    def render(self):
        return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__delattr__('move')


class Door(Entity):

    @property
    def var_is_blocking_pos(self):
        return False if self.is_open else True

    @property
    def var_is_blocking_light(self):
        return False if self.is_open else True

    @property
    def var_can_collide(self):
        return False if self.is_open else True

    @property
    def encoding(self):
        return d.VALUE_CLOSED_DOOR if self.is_closed else d.VALUE_OPEN_DOOR

    @property
    def str_state(self):
        return 'open' if self.is_open else 'closed'

    def __init__(self, *args, closed_on_init=True, auto_close_interval=10, indicate_area=False, **kwargs):
        self._status = d.STATE_CLOSED
        super(Door, self).__init__(*args, **kwargs)
        self.auto_close_interval = auto_close_interval
        self.time_to_close = 0
        if not closed_on_init:
            self._open()
        else:
            self._close()
        if indicate_area:
            self._collection.add_items([DoorIndicator(x) for x in self.tile.neighboring_floor])

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(state=str(self.str_state), time_to_close=int(self.time_to_close))
        return state_dict

    @property
    def is_closed(self):
        return self._status == d.STATE_CLOSED

    @property
    def is_open(self):
        return self._status == d.STATE_OPEN

    @property
    def status(self):
        return self._status

    def render(self):
        name, state = 'door_open' if self.is_open else 'door_closed', 'blank'
        return RenderEntity(name, self.pos, 1, 'none', state, self.u_int + 1)

    def use(self):
        if self._status == d.STATE_OPEN:
            self._close()
        else:
            self._open()
        return c.VALID

    def tick(self):
        if self.is_open and len(self.tile) == 1 and self.time_to_close:
            self.time_to_close -= 1
            return c.NOT_VALID
        elif self.is_open and not self.time_to_close and len(self.tile) == 1:
            self.use()
            return c.VALID
        else:
            return c.NOT_VALID

    def _open(self):
        self._status = d.STATE_OPEN
        self.time_to_close = self.auto_close_interval

    def _close(self):
        self._status = d.STATE_CLOSED
