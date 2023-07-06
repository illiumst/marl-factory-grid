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
        return self._tile.pos

    @property
    def tile(self):
        return self._tile

    @property
    def last_tile(self):
        try:
            return self._last_tile
        except AttributeError:
            # noinspection PyAttributeOutsideInit
            self._last_tile = None
            return self._last_tile

    @property
    def last_pos(self):
        try:
            return self.last_tile.pos
        except AttributeError:
            return c.VALUE_NO_POS

    @property
    def direction_of_view(self):
        last_x, last_y = self.last_pos
        curr_x, curr_y = self.pos
        return last_x - curr_x, last_y - curr_y

    def move(self, next_tile):
        curr_tile = self.tile
        if not_same_tile := curr_tile != next_tile:
            if valid := next_tile.enter(self):
                curr_tile.leave(self)
                self._tile = next_tile
                self._last_tile = curr_tile
                for observer in self.observers:
                    observer.notify_change_pos(self)
            return valid
        return not_same_tile

    def __init__(self, tile, **kwargs):
        super().__init__(**kwargs)
        self._status = None
        self._tile = tile
        tile.enter(self)

    def summarize_state(self) -> dict:
        return dict(name=str(self.name), x=int(self.x), y=int(self.y),
                    tile=str(self.tile.name), can_collide=bool(self.var_can_collide))

    @abc.abstractmethod
    def render(self):
        return RenderEntity(self.__class__.__name__.lower(), self.pos)

    def __repr__(self):
        return super(Entity, self).__repr__() + f'(@{self.pos})'
