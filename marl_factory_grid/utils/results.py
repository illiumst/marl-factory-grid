from typing import Union
from dataclasses import dataclass

from marl_factory_grid.environment.entity.object import Object

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
    reward: Union[float, None] = None
    value: Union[float, None] = None
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
    """
    A specific Result class representing outcomes of actions.
    """
    pass


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
