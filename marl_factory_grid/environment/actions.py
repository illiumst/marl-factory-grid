import abc
from typing import Union

from marl_factory_grid.environment import rewards as r, constants as c
from marl_factory_grid.utils.helpers import MOVEMAP
from marl_factory_grid.utils.results import ActionResult


class Action(abc.ABC):

    @property
    def name(self):
        return self._identifier

    @abc.abstractmethod
    def __init__(self, identifier: str):
        self._identifier = identifier

    @abc.abstractmethod
    def do(self, entity, state) -> Union[None, ActionResult]:
        print()
        return

    def __repr__(self):
        return f'Action[{self._identifier}]'


class Noop(Action):

    def __init__(self):
        super().__init__(c.NOOP)

    def do(self, entity, *_) -> Union[None, ActionResult]:
        return ActionResult(identifier=self._identifier, validity=c.VALID,
                            reward=r.NOOP, entity=entity)


class Move(Action, abc.ABC):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do(self, entity, state):
        new_pos = self._calc_new_pos(entity.pos)
        if state.check_move_validity(entity, new_pos):
            # noinspection PyUnresolvedReferences
            move_validity = entity.move(new_pos, state)
            reward = r.MOVEMENTS_VALID if move_validity else r.MOVEMENTS_FAIL
            return ActionResult(entity=entity, identifier=self._identifier, validity=move_validity, reward=reward)
        else:  # There is no place to go, propably collision
            # This is currently handeld by the WatchCollisions rule, so that it can be switched on and off by conf.yml
            # return ActionResult(entity=entity, identifier=self._identifier, validity=c.NOT_VALID, reward=r.COLLISION)
            return ActionResult(entity=entity, identifier=self._identifier, validity=c.NOT_VALID)

    def _calc_new_pos(self, pos):
        x_diff, y_diff = MOVEMAP[self._identifier]
        return pos[0] + x_diff, pos[1] + y_diff


class North(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.NORTH, *args, **kwargs)


class NorthEast(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.NORTHEAST, *args, **kwargs)


class East(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.EAST, *args, **kwargs)


class SouthEast(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.SOUTHEAST, *args, **kwargs)


class South(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.SOUTH, *args, **kwargs)


class SouthWest(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.SOUTHWEST, *args, **kwargs)


class West(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.WEST, *args, **kwargs)


class NorthWest(Move):
    def __init__(self, *args, **kwargs):
        super().__init__(c.NORTHWEST, *args, **kwargs)


Move4 = [North, East, South, West]
# noinspection PyTypeChecker
Move8 = Move4 + [NorthEast, SouthEast, SouthWest, NorthWest]

ALL_BASEACTIONS = Move8 + [Noop]
