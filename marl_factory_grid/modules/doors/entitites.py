from typing import Union

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils import Result
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.doors import constants as d


class DoorIndicator(Entity):

    @property
    def encoding(self):
        return d.VALUE_ACCESS_INDICATOR

    def render(self):
        return []

    def __init__(self, *args, **kwargs):
        """
        Is added around a door for agents to see.
        """
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
    def str_state(self) -> str:
        """
        Internal Usage
        """
        return 'open' if self.is_open else 'closed'

    @property
    def is_closed(self) -> bool:
        return self._state == d.STATE_CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == d.STATE_OPEN

    @property
    def time_to_close(self):
        """
        :returns: The time it takes for the door to close.
        :rtype: float
        """
        return self._time_to_close

    def __init__(self, *args, closed_on_init=True, auto_close_interval=10, **kwargs):
        """
        A door entity that can be opened or closed by agents or rules.

        :param closed_on_init: Whether the door spawns as open or closed.
        :type closed_on_init: bool

        :param auto_close_interval: after how many steps should the door automatically close itself,
        :type auto_close_interval: int
        """
        self._state = d.STATE_CLOSED
        super(Door, self).__init__(*args, **kwargs)
        self._auto_close_interval = auto_close_interval
        self._time_to_close = 0
        if not closed_on_init:
            self._open()
        else:
            self._close()

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(state=str(self.str_state), time_to_close=self.time_to_close)
        return state_dict

    def render(self):
        name, state = 'door_open' if self.is_open else 'door_closed', 'blank'
        return RenderEntity(name, self.pos, 1, 'none', state, self.u_int + 1)

    def use(self) -> bool:
        """
        Internal usage
        """
        if self._state == d.STATE_OPEN:
            self._close()
        else:
            self._open()
        return c.VALID

    def tick(self, state) -> Union[Result, None]:
        # Check if no entity is standing in the door
        if len(state.entities.pos_dict[self.pos]) <= 2:
            if self.is_open and self.time_to_close:
                self._decrement_timer()
                return Result(f"{d.DOOR}_tick", c.VALID, entity=self)
            elif self.is_open and not self.time_to_close:
                self.use()
                return Result(f"{d.DOOR}_closed", c.VALID, entity=self)
            else:
                # No one is in door, but it is closed... Nothing to do....
                return None
        else:
            # Entity is standing in the door, reset timer
            self._reset_timer()
            return Result(f"{d.DOOR}_reset", c.VALID, entity=self)

    def _open(self) -> bool:
        """
        Internal Usage
        """
        self._state = d.STATE_OPEN
        self._reset_timer()
        return True

    def _close(self) -> bool:
        """
        Internal Usage
        """
        self._state = d.STATE_CLOSED
        return True

    def _decrement_timer(self) -> bool:
        """
        Internal Usage
        """
        self._time_to_close -= 1
        return True

    def _reset_timer(self) -> bool:
        """
        Internal Usage
        """
        self._time_to_close = self._auto_close_interval
        return True

    def reset(self):
        """
        Internal Usage
        """
        self._close()
        self._reset_timer()
