from typing import Union
from dataclasses import dataclass

from marl_factory_grid.environment.entity.object import Object
import marl_factory_grid.environment.constants as c
TYPE_VALUE = 'value'
TYPE_REWARD = 'reward'


TYPES = [TYPE_VALUE, TYPE_REWARD]


@dataclass
class InfoObject:
    """
    Data class representing information about an entity or the global environment.
    """
    identifier: str
    val_type: str
    value: Union[float, int]


@dataclass
class Result:
    """
    A generic result class representing outcomes of operations or actions.

    Attributes:
        - identifier: A unique identifier for the result.
        - validity: A boolean indicating whether the operation or action was successful.
        - reward: The reward associated with the result, if applicable.
        - value: The value associated with the result, if applicable.
        - entity: The entity associated with the result, if applicable.
    """
    identifier: str
    validity: bool
    reward: float | None = None
    value: float | None = None
    collision: bool | None = None
    entity: Object = None

    def get_infos(self):
        """
        Get information about the result.

        :return: A list of InfoObject representing different types of information.
        """
        n = self.entity.name if self.entity is not None else "Global"
        # Return multiple Info Dicts
        return [InfoObject(identifier=f'{n}_{self.identifier}',
                           val_type=t, value=self.__getattribute__(t)) for t in TYPES
                if self.__getattribute__(t) is not None]

    def __repr__(self):
        valid = "not " if not self.validity else ""
        reward = f" | Reward: {self.reward}" if self.reward is not None else ""
        value = f" | Value: {self.value}" if self.value is not None else ""
        entity = f" | by: {self.entity.name}" if self.entity is not None else ""
        return f'{self.__class__.__name__}({self.identifier.capitalize()} {valid}valid{reward}{value}{entity})'


@dataclass
class ActionResult(Result):
    def __init__(self, *args, action_introduced_collision: bool = False, **kwargs):
        """
        A specific Result class representing outcomes of actions.

        :param action_introduced_collision: Wether the action did introduce a colision between agents or other entities.
                                            These need to be able to collide.
        """
        super().__init__(*args, **kwargs)
        self.action_introduced_collision = action_introduced_collision

    def __repr__(self):
        sr = super().__repr__()
        return sr + f" | {c.COLLISION}" if self.action_introduced_collision is not None else ""

    def get_infos(self):
        base_infos = super().get_infos()
        if self.action_introduced_collision:
            i = InfoObject(identifier=f'{self.entity.name}_{c.COLLISION}', val_type=TYPE_VALUE, value=1)
            return base_infos + [i]
        else:
            return base_infos

@dataclass
class DoneResult(Result):
    """
    A specific Result class representing the completion of an action or operation.
    """
    pass


@dataclass
class State(Result):
    # TODO: change identifier to action/last_action
    pass

@dataclass
class TickResult(Result):
    """
    A specific Result class representing outcomes of tick operations.
    """
    pass
