from marl_factory_grid.environment.entity.entity import Entity
from ...utils.utility_classes import RenderEntity
from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.results import TickResult

from . import constants as m


class Machine(Entity):

    @property
    def encoding(self):
        return self._encodings[self.status]

    def __init__(self, *args, work_interval: int = 10, pause_interval: int = 15, **kwargs):
        """
        Represents a machine entity that the maintainer will try to maintain.

        :param work_interval: How long should the machine work before pausing.
        :type work_interval: int
        :param pause_interval: How long should the machine pause before continuing to work.
        :type pause_interval: int
        """
        super(Machine, self).__init__(*args, **kwargs)
        self._intervals = dict({m.STATE_IDLE: pause_interval, m.STATE_WORK: work_interval})
        self._encodings = dict({m.STATE_IDLE: pause_interval, m.STATE_WORK: work_interval})

        self.status = m.STATE_IDLE
        self.health = 100
        self._counter = 0

    def maintain(self) -> bool:
        """
        Attempts to maintain the machine by increasing its health.
        """
        if self.status == m.STATE_WORK:
            return c.NOT_VALID
        if self.health <= 98:
            self.health = 100
            return c.VALID
        else:
            return c.NOT_VALID

    def tick(self, state):
        """
        Updates the machine's mode (work, pause) depending on its current counter and whether an agent is currently on
        its position. If no agent is standing on the machine's position, it decrements its own health.

        :param state: The current game state.
        :type state: GameState
        :return: The result of the tick operation on the machine.
        :rtype: TickResult | None
        """
        others = state.entities.pos_dict[self.pos]
        if self.status == m.STATE_MAINTAIN and any([c.AGENT in x.name for x in others]):
            return TickResult(identifier=self.name, validity=c.VALID, entity=self)
        elif self.status == m.STATE_MAINTAIN and not any([c.AGENT in x.name for x in others]):
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
        """
        Internal Usage
        """
        self._counter = self._intervals[self.status]

    def render(self):
        return RenderEntity(m.MACHINE, self.pos)
