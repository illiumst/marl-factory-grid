from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult

from . import constants as m


class Machine(Entity):

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

    @property
    def encoding(self):
        return self._encodings[self.status]

    def __init__(self, *args, work_interval: int = 10, pause_interval: int = 15, **kwargs):
        super(Machine, self).__init__(*args, **kwargs)
        self._intervals = dict({m.STATE_IDLE: pause_interval, m.STATE_WORK: work_interval})
        self._encodings = dict({m.STATE_IDLE: pause_interval, m.STATE_WORK: work_interval})

        self.status = m.STATE_IDLE
        self.health = 100
        self._counter = 0

    def maintain(self):
        if self.status == m.STATE_WORK:
            return c.NOT_VALID
        if self.health <= 98:
            self.health = 100
            return c.VALID
        else:
            return c.NOT_VALID

    def tick(self):
        if self.status == m.STATE_MAINTAIN and any([c.AGENT in x.name for x in self.tile.guests]):
            return TickResult(identifier=self.name, validity=c.VALID, reward=0, entity=self)
        elif self.status == m.STATE_MAINTAIN and not any([c.AGENT in x.name for x in self.tile.guests]):
            self.status = m.STATE_WORK
            self.reset_counter()
            return None
        elif self._counter:
            self._counter -= 1
            self.health -= 1
            return None
        else:
            self.status = m.STATE_WORK if self.status == m.STATE_IDLE else m.STATE_IDLE
            self.reset_counter()
            return None

    def reset_counter(self):
        self._counter = self._intervals[self.status]

    def render(self):
        return RenderEntity(m.MACHINE, self.pos)
