from typing import Dict, List, Union

import numpy as np

from environments.factory.base.objects import Agent, Entity, Action
from environments.factory.factory_dirt import Dirt, DirtRegister, DirtFactory
from environments.factory.base.objects import Floor
from environments.factory.base.registers import Floors, Entities, EntityRegister


class Machines(EntityRegister):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Machine(Entity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StationaryMachinesDirtFactory(DirtFactory):

    def __init__(self, *args, **kwargs):
        self._machine_coords = [(6, 6), (12, 13)]
        super().__init__(*args, **kwargs)

    def entities_hook(self) -> Dict[(str, Entities)]:
        super_entities = super().entities_hook()

        return super_entities

    def reset_hook(self) -> None:
                pass

    def observations_hook(self) -> Dict[str, np.typing.ArrayLike]:
        pass

    def actions_hook(self) -> Union[Action, List[Action]]:
        pass

    def step_hook(self) -> (List[dict], dict):

        pass

    def per_agent_raw_observations_hook(self, agent) -> Dict[str, np.typing.ArrayLike]:
        super_per_agent_raw_observations = super().per_agent_raw_observations_hook(agent)
        return super_per_agent_raw_observations

    def per_agent_reward_hook(self, agent: Agent) -> Dict[str, dict]:
        pass

    def pre_step_hook(self) -> None:
        pass

    def post_step_hook(self) -> dict:
        pass
