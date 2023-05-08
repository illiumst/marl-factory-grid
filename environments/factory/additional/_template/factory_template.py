import time
from typing import List, Union, Dict
import random

import numpy as np

from environments.factory.additional.doors.doors_collections import Doors
from environments.factory.additional.doors.doors_util import DoorProperties, RewardsDoor, Constants, Actions
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action
from environments.factory.base.registers import Entities

from environments import helpers as h

from environments.factory.base.renderer import RenderEntity
from environments.utility_classes import ObservationProperties


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def entropy(x):
    return -(x * np.log(x + 1e-8)).sum()


c = Constants
a = Actions


# noinspection PyAttributeOutsideInit, PyAbstractClass
class DoorFactory(BaseFactory):

    @property
    def actions_hook(self) -> Union[Action, List[Action]]:
        super_actions = super().actions_hook
        super_actions.append(Action(str_ident=a.USE_DOOR))
        return super_actions

    @property
    def entities_hook(self) -> Dict[(str, Entities)]:
        super_entities = super().entities_hook

        parsed_doors = h.one_hot_level(self._parsed_level, c.DOOR_SYMBOL)
        parsed_doors = np.pad(parsed_doors, self.obs_prop.pomdp_r, 'constant', constant_values=0)
        if np.any(parsed_doors):
            door_tiles = [self[c.FLOOR].by_pos(tuple(pos)) for pos in np.argwhere(parsed_doors == c.OCCUPIED_CELL)]
            doors = Doors.from_tiles(door_tiles, self._level_shape, indicate_area=self.obs_prop.indicate_door_area,
                                     entity_kwargs=dict()
                                     )
            super_entities.update(({c.DOORS: doors}))
        return super_entities

    def __init__(self, *args,
                 door_properties: DoorProperties = DoorProperties(), rewards_door: RewardsDoor = RewardsDoor(),
                 env_seed=time.time_ns(), **kwargs):
        if isinstance(door_properties, dict):
            door_properties = DoorProperties(**door_properties)
        if isinstance(rewards_door, dict):
            rewards_door = RewardsDoor(**rewards_door)
        self.door_properties = door_properties
        self.rewards_door = rewards_door
        self._door_rng = np.random.default_rng(env_seed)
        self._doors: Doors
        kwargs.update(env_seed=env_seed)
        # TODO: Reset ---> document this
        super().__init__(*args, **kwargs)

    def render_assets_hook(self, mode='human'):
        additional_assets = super().render_assets_hook()
        doors = []
        for i, door in enumerate(self[c.DOORS]):
            name, state = 'door_open' if door.is_open else 'door_closed', 'blank'
            doors.append(RenderEntity(name, door.pos, 1, 'none', state, i + 1))
        additional_assets.extend(doors)
        return additional_assets


    def step_hook(self) -> (List[dict], dict):
        super_reward_info = super().step_hook()
        # Step the door close intervall
        # TODO: Maybe move this to self.post_step_hook? May collide with reward calculation.
        if doors := self[c.DOORS]:
            doors.tick_doors()
        return super_reward_info

    def do_additional_actions(self, agent: Agent, action: Action) -> (dict, dict):
        action_result = super().do_additional_actions(agent, action)
        if action_result is None:
            if action == a.USE_DOOR:
                return self.use_door_action(agent)
            else:
                return None
        else:
            return action_result

    def use_door_action(self, agent: Agent):

        # Check if agent really is standing on a door:
        door = self[c.DOORS].get_near_position(agent.pos)
        if door is not None:
            door.use()
            valid = c.VALID
            self.print(f'{agent.name} just used a {door.name} at {door.pos}')
            info_dict = {f'{agent.name}_door_use': 1, f'door_use': 1}
        # When he doesn't...
        else:
            valid = c.NOT_VALID
            info_dict = {f'{agent.name}_failed_door_use': 1, 'failed_door_use': 1}
            self.print(f'{agent.name} just tried to use a door at {agent.pos}, but there is none.')

        reward = dict(value=self.rewards_door.USE_DOOR_VALID if valid else self.rewards_door.USE_DOOR_FAIL,
                      reason=a.USE_DOOR, info=info_dict)

        return valid, reward

    def reset_hook(self) -> None:
        super().reset_hook()
        # There is nothing to reset.

    def check_additional_done(self) -> (bool, dict):
        super_done, super_dict = super().check_additional_done()
        return super_done, super_dict

    def observations_hook(self) -> Dict[str, np.typing.ArrayLike]:
        additional_observations = super().observations_hook()

        additional_observations.update({c.DOORS: self[c.DOORS].as_array()})
        return additional_observations

    def post_step_hook(self) -> List[Dict[str, int]]:
        super_post_step = super(DoorFactory, self).post_step_hook()
        return super_post_step


if __name__ == '__main__':
    from environments.utility_classes import AgentRenderOptions as aro
    render = True

    door_props = DoorProperties(
        indicate_door_area=True
    )

    obs_props = ObservationProperties(render_agents=aro.COMBINED, omit_agent_self=True,
                                      pomdp_r=2, additional_agent_placeholder=None, cast_shadows=True
                                      )

    move_props = {'allow_square_movement': True,
                  'allow_diagonal_movement': False,
                  'allow_no_op': False}
    import time
    global_timings = []
    for i in range(10):

        factory = DoorFactory(n_agents=10, done_at_collision=False,
                              level_name='rooms', max_steps=1000,
                              obs_prop=obs_props, parse_doors=True,
                              verbose=True,
                              mv_prop=move_props, dirt_prop=door_props,
                              # inject_agents=[TSPDirtAgent],
                              )

        # noinspection DuplicatedCode
        n_actions = factory.action_space.n - 1
        _ = factory.observation_space
        obs_space = factory.observation_space
        obs_space_named = factory.named_observation_space
        action_space_named = factory.named_action_space
        times = []
        for epoch in range(10):
            start_time = time.time()
            random_actions = [[random.randint(0, n_actions) for _
                               in range(factory.n_agents)] for _
                              in range(factory.max_steps+1)]
            env_state = factory.reset()
            if render:
                factory.render()
            # tsp_agent = factory.get_injected_agents()[0]

            rwrd = 0
            for agent_i_action in random_actions:
                # agent_i_action = tsp_agent.predict()
                env_state, step_rwrd, done_bool, info_obj = factory.step(agent_i_action)
                rwrd += step_rwrd
                if render:
                    factory.render()
                if done_bool:
                    break
            times.append(time.time() - start_time)
            # print(f'Factory run {epoch} done, reward is:\n    {r}')
        print('Mean Time Taken: ', sum(times) / 10)
        global_timings.extend(times)
    print('Mean Time Taken: ', sum(global_timings) / len(global_timings))
    print('Median Time Taken: ', global_timings[len(global_timings)//2])
