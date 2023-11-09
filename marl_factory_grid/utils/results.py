from typing import Union
from dataclasses import dataclass

TYPE_VALUE  = 'value'
TYPE_REWARD = 'reward'
types = [TYPE_VALUE, TYPE_REWARD]

@dataclass
class InfoObject:
    identifier: str
    val_type: str
    value: Union[float, int]


@dataclass
class Result:
    identifier: str
    validity: bool
    reward: Union[float, None] = None
    value: Union[float, None] = None
    entity: None = None

    def get_infos(self):
        n = self.entity.name if self.entity is not None else "Global"
        return [InfoObject(identifier=f'{n}_{self.identifier}_{t}',
                           val_type=t, value=self.__getattribute__(t)) for t in types
                if self.__getattribute__(t) is not None]

    def __repr__(self):
        valid = "not " if not self.validity else ""
        reward = f" | Reward: {self.reward}" if self.reward is not None else ""
        value = f" | Value: {self.value}" if self.value is not None else ""
        entity = f" | by: {self.entity.name}" if self.entity is not None else ""
        return f'{self.__class__.__name__}({self.identifier.capitalize()} {valid}valid{reward}{value})'


@dataclass
class TickResult(Result):
    pass


@dataclass
class ActionResult(Result):
    pass


@dataclass
class DoneResult(Result):
    pass
