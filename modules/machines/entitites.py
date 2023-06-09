from environment.entity.entity import Entity
from environment.utils.render import RenderEntity
from environment import constants as c
from environment.utils.results import TickResult
from modules.machines import constants as m, rewards as r


class Machine(Entity):

    @property
    def encoding(self):
        return self._encodings[self.state]

    def __init__(self, *args, work_interval: int = 10, pause_interval: int = 15, **kwargs):
        super(Machine, self).__init__(*args, **kwargs)
        self._intervals = dict({m.STATE_IDLE: pause_interval, m.STATE_WORK: work_interval})
        self._encodings = dict({m.STATE_IDLE: pause_interval, m.STATE_WORK: work_interval})

        self.state = m.STATE_IDLE
        self.health = 100
        self._counter = 0
        self.__delattr__('move')

    def maintain(self):
        if self.state == m.STATE_WORK:
            return c.NOT_VALID
        if self.health <= 98:
            self.health = 100
            return c.VALID
        else:
            return c.NOT_VALID

    def tick(self):
        if self.state == m.STATE_MAINTAIN and any([c.AGENT in x.name for x in self.tile.guests]):
            return TickResult(self.name, c.VALID, r.NONE, self)
        elif self.state == m.STATE_MAINTAIN and not any([c.AGENT in x.name for x in self.tile.guests]):
            self.state = m.STATE_WORK
            self.reset_counter()
            return None
        elif self._counter:
            self._counter -= 1
            self.health -= 1
            return None
        else:
            self.state = m.STATE_WORK if self.state == m.STATE_IDLE else m.STATE_IDLE
            self.reset_counter()
            return None

    def reset_counter(self):
        self._counter = self._intervals[self.state]

    def render(self):
        return RenderEntity(m.MACHINE, self.pos)
