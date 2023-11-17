import abc
import random
from typing import Union

from marl_factory_grid.environment import rewards as r, constants as c
from marl_factory_grid.utils.helpers import MOVEMAP
from marl_factory_grid.utils.results import ActionResult


class Action(abc.ABC):

    @property
    def name(self):
        return self._identifier

    @abc.abstractmethod
    def __init__(self, identifier: str, default_valid_reward: float,  default_fail_reward: float,
                 valid_reward: float | None = None, fail_reward: float | None = None):
        self.fail_reward = fail_reward if fail_reward is not None else default_fail_reward
        self.valid_reward = valid_reward if valid_reward is not None else default_valid_reward
        self._identifier = identifier

    @abc.abstractmethod
    def do(self, entity, state) -> Union[None, ActionResult]:
        validity = bool(random.choice([0, 1]))
        return self.get_result(validity, entity)

    def __repr__(self):
        return f'Action[{self._identifier}]'

    def get_result(self, validity, entity):
        reward = self.valid_reward if validity else self.fail_reward
        return ActionResult(self.__class__.__name__, validity, reward=reward, entity=entity)


class Noop(Action):

    def __init__(self, **kwargs):
        super().__init__(c.NOOP, r.NOOP, r.NOOP, **kwargs)

    def do(self, entity, *_) -> Union[None, ActionResult]:
        return self.get_result(c.VALID, entity)


class Move(Action, abc.ABC):

    @abc.abstractmethod
    def __init__(self, identifier, **kwargs):
        super().__init__(identifier, r.MOVEMENTS_VALID, r.MOVEMENTS_FAIL, **kwargs)

    def do(self, entity, state):
        new_pos = self._calc_new_pos(entity.pos)
        if state.check_move_validity(entity, new_pos):
            valid = entity.move(new_pos, state)
            # Aftermath Collision Check
            if len([x for x in state.entities.by_pos(entity.pos) if x.var_can_collide]) > 1:
                # The entity did move, but there was something to collide with...
                #  Is then reported as a non-valid move, which did work.
                valid = False

        else:
            # There is no place to go, propably collision
            # This is currently handeld by the WatchCollisions rule, so that it can be switched on and off by conf.yml
            # return ActionResult(entity=entity, identifier=self._identifier, validity=c.NOT_VALID, reward=r.COLLISION)
            valid = c.NOT_VALID
        if valid:
            state.print(f'{entity.name} just moved to {entity.pos}.')
        else:
            state.print(f'{entity.name} just tried to move to {new_pos} but either failed or hat a Collision.')
        return self.get_result(valid, entity)

    def _calc_new_pos(self, pos):
        x_diff, y_diff = MOVEMAP[self._identifier]
        return pos[0] + x_diff, pos[1] + y_diff


class North(Move):
    def __init__(self, **kwargs):
        super().__init__(c.NORTH, **kwargs)


class NorthEast(Move):
    def __init__(self, **kwargs):
        super().__init__(c.NORTHEAST, **kwargs)


class East(Move):
    def __init__(self, **kwargs):
        super().__init__(c.EAST, **kwargs)


class SouthEast(Move):
    def __init__(self, **kwargs):
        super().__init__(c.SOUTHEAST, **kwargs)


class South(Move):
    def __init__(self, **kwargs):
        super().__init__(c.SOUTH, **kwargs)


class SouthWest(Move):
    def __init__(self, **kwargs):
        super().__init__(c.SOUTHWEST, **kwargs)


class West(Move):
    def __init__(self, **kwargs):
        super().__init__(c.WEST, **kwargs)


class NorthWest(Move):
    def __init__(self, **kwargs):
        super().__init__(c.NORTHWEST, **kwargs)


Move4 = [North, East, South, West]
# noinspection PyTypeChecker
Move8 = Move4 + [NorthEast, SouthEast, SouthWest, NorthWest]

ALL_BASEACTIONS = Move8 + [Noop]
