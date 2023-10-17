import abc

from .. import constants as c
from .object import EnvObject
from ...utils.render import RenderEntity
from ...utils.results import ActionResult


class Entity(EnvObject, abc.ABC):
    """Full Env Entity that lives on the environment Grid. Doors, Items, DirtPile etc..."""

    @property
    def state(self):
        return self._status or ActionResult(entity=self, identifier=c.NOOP, validity=c.VALID, reward=0)

    @property
    def var_has_position(self):
        return self.pos != c.VALUE_NO_POS

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def pos(self):
        return self._pos

    @property
    def tile(self):
        return self._tile  # wall_n_floors funktionalität

    # @property
    # def last_tile(self):
    #     try:
    #         return self._last_tile
    #     except AttributeError:
    #         # noinspection PyAttributeOutsideInit
    #         self._last_tile = None
    #         return self._last_tile

    @property
    def direction_of_view(self):
        last_x, last_y = self._last_pos
        curr_x, curr_y = self.pos
        return last_x - curr_x, last_y - curr_y

    def move(self, next_pos, state):
        next_pos = next_pos
        curr_pos = self._pos
        if not_same_pos := curr_pos != next_pos:
            if valid := state.check_move_validity(self, next_pos):
                self._pos = next_pos
                self._last_pos = curr_pos
                for observer in self.observers:
                    observer.notify_change_pos(self)
            return valid
        return not_same_pos

    def __init__(self, pos, bind_to=None, **kwargs):
        super().__init__(**kwargs)
        self._status = None
        self._pos = pos
        if bind_to:
            try:
                self.bind_to(bind_to)
            except AttributeError:
                print(f'Objects of {self.__class__.__name__} can not be bound to other entities.')
                exit()

    def summarize_state(self) -> dict:  # tile=str(self.tile.name)
        return dict(name=str(self.name), x=int(self.x), y=int(self.y), can_collide=bool(self.var_can_collide))

    @abc.abstractmethod
    def render(self):
        return RenderEntity(self.__class__.__name__.lower(), self.pos)

    def __repr__(self):
        return super(Entity, self).__repr__() + f'(@{self.pos})'
