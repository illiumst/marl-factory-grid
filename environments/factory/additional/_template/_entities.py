from environments.factory.base.objects import Entity
from environments.factory.additional.doors.doors_util import Constants as c


class Template(Entity):
    """Template for new Entity"""

    # How to define / override properties
    @property
    def is_blocking(self):
        return False

    @property
    def can_collide(self):
        return False if self.template_attr else True

    @property
    def encoding(self):
        # This is important as it shadow is checked by occupation value
        return c.CLOSED_DOOR_CELL if self.is_closed else c.OPEN_DOOR_CELL

    @property
    def str_state(self):
        return 'open' if self.is_open else 'closed'

    def __init__(self, *args, closed_on_init=True, auto_close_interval=10, indicate_area=False, **kwargs):
        super(Template, self).__init__(*args, **kwargs)
        self._state = c.CLOSED_DOOR
        self.indicate_area = indicate_area
        self.auto_close_interval = auto_close_interval
        self.time_to_close = -1
        if not closed_on_init:
            self._open()

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(state=str(self.str_state), time_to_close=int(self.time_to_close))
        return state_dict

    @property
    def is_closed(self):
        return self._state == c.CLOSED_DOOR

    @property
    def is_open(self):
        return self._state == c.OPEN_DOOR

    @property
    def status(self):
        return self._state

    def use(self):
        if self._state == c.OPEN_DOOR:
            self._close()
        else:
            self._open()

    def tick(self):
        if self.is_open and len(self.tile) == 1 and self.time_to_close:
            self.time_to_close -= 1
        elif self.is_open and not self.time_to_close and len(self.tile) == 1:
            self.use()

    def _open(self):
        self._state = c.OPEN_DOOR
        self._collection.notify_change_to_value(self)
        self.time_to_close = self.auto_close_interval

    def _close(self):
        self._state = c.CLOSED_DOOR
        self._collection.notify_change_to_value(self)
