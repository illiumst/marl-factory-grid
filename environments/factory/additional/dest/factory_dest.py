import time
from enum import Enum
from typing import List, Union, Dict
import numpy as np
import random

from environments.factory.additional.dest.dest_collections import Destinations, ReachedDestinations
from environments.factory.additional.dest.dest_enitites import Destination
from environments.factory.additional.dest.dest_util import Constants, Actions, RewardsDest, DestModeOptions, \
    DestProperties
from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action
from environments.factory.base.registers import Entities

from environments.factory.base.renderer import RenderEntity

c = Constants
a = Actions


# noinspection PyAttributeOutsideInit, PyAbstractClass
class DestFactory(BaseFactory):
    # noinspection PyMissingConstructor

    def __init__(self, *args, dest_prop: DestProperties = DestProperties(), rewards_dest: RewardsDest = RewardsDest(),
                 env_seed=time.time_ns(), **kwargs):
        if isinstance(dest_prop, dict):
            dest_prop = DestProperties(**dest_prop)
        if isinstance(rewards_dest, dict):
            rewards_dest = RewardsDest(**rewards_dest)
        self.dest_prop = dest_prop
        self.rewards_dest = rewards_dest
        kwargs.update(env_seed=env_seed)
        self._dest_rng = np.random.default_rng(env_seed)
        super().__init__(*args, **kwargs)

    @property
    def actions_hook(self) -> Union[Action, List[Action]]:
        # noinspection PyUnresolvedReferences
        super_actions = super().actions_hook
        # If targets are considers reached after some time, agents need an action for that.
        if self.dest_prop.dwell_time:
            super_actions.append(Action(enum_ident=a.WAIT_ON_DEST))
        return super_actions

    @property
    def entities_hook(self) -> Dict[(Enum, Entities)]:
        # noinspection PyUnresolvedReferences
        super_entities = super().entities_hook

        empty_tiles = self[c.FLOOR].empty_tiles[:self.dest_prop.n_dests]
        destinations = Destinations.from_tiles(
            empty_tiles, self._level_shape,
            entity_kwargs=dict(
                dwell_time=self.dest_prop.dwell_time)
        )
        reached_destinations = ReachedDestinations(level_shape=self._level_shape)

        super_entities.update({c.DEST: destinations, c.DEST_REACHED: reached_destinations})
        return super_entities

    def do_wait_action(self, agent: Agent) -> (dict, dict):
        if destination := self[c.DEST].by_pos(agent.pos):
            valid = destination.do_wait_action(agent)
            self.print(f'{agent.name} just waited at {agent.pos}')
            info_dict = {f'{agent.name}_{a.WAIT_ON_DEST}_VALID': 1}
        else:
            valid = c.NOT_VALID
            self.print(f'{agent.name} just tried to do_wait_action do_wait_action at {agent.pos} but failed')
            info_dict = {f'{agent.name}_{a.WAIT_ON_DEST}_FAIL': 1}
        reward = dict(value=self.rewards_dest.WAIT_VALID if valid else self.rewards_dest.WAIT_FAIL,
                      reason=a.WAIT_ON_DEST, info=info_dict)
        return valid, reward

    def do_additional_actions(self, agent: Agent, action: Action) -> (dict, dict):
        # noinspection PyUnresolvedReferences
        super_action_result = super().do_additional_actions(agent, action)
        if super_action_result is None:
            if action == a.WAIT_ON_DEST:
                action_result = self.do_wait_action(agent)
                return action_result
            else:
                return None
        else:
            return super_action_result

    def reset_hook(self) -> None:
        # noinspection PyUnresolvedReferences
        super().reset_hook()
        self._dest_spawn_timer = dict()

    def trigger_destination_spawn(self):
        destinations_to_spawn = [key for key, val in self._dest_spawn_timer.items()
                                 if val == self.dest_prop.spawn_frequency]
        if destinations_to_spawn:
            n_dest_to_spawn = len(destinations_to_spawn)
            if self.dest_prop.spawn_mode != DestModeOptions.GROUPED:
                destinations = [Destination(tile, self[c.DEST]) for tile in self[c.FLOOR].empty_tiles[:n_dest_to_spawn]]
                self[c.DEST].add_additional_items(destinations)
                for dest in destinations_to_spawn:
                    del self._dest_spawn_timer[dest]
                self.print(f'{n_dest_to_spawn} new destinations have been spawned')
            elif self.dest_prop.spawn_mode == DestModeOptions.GROUPED and n_dest_to_spawn == self.dest_prop.n_dests:
                destinations = [Destination(tile, self[c.DEST]) for tile in self[c.FLOOR].empty_tiles[:n_dest_to_spawn]]
                self[c.DEST].add_additional_items(destinations)
                for dest in destinations_to_spawn:
                    del self._dest_spawn_timer[dest]
                self.print(f'{n_dest_to_spawn} new destinations have been spawned')
            else:
                self.print(f'{n_dest_to_spawn} new destinations could be spawned, but waiting for all.')
                pass
        else:
            self.print('No Items are spawning, limit is reached.')

    def step_hook(self) -> (List[dict], dict):
        # noinspection PyUnresolvedReferences
        super_reward_info = super().step_hook()
        for key, val in self._dest_spawn_timer.items():
            self._dest_spawn_timer[key] = min(self.dest_prop.spawn_frequency, self._dest_spawn_timer[key] + 1)

        for dest in list(self[c.DEST].values()):
            if dest.is_considered_reached:
                dest.change_parent_collection(self[c.DEST_REACHED])
                self._dest_spawn_timer[dest.name] = 0
                self.print(f'{dest.name} is reached now, removing...')
            else:
                for agent_name in dest.currently_dwelling_names:
                    agent = self[c.AGENT].by_name(agent_name)
                    if agent.pos == dest.pos:
                        self.print(f'{agent.name} is still waiting.')
                        pass
                    else:
                        dest.leave(agent)
                        self.print(f'{agent.name} left the destination early.')
        self.trigger_destination_spawn()
        return super_reward_info

    def observations_hook(self) -> Dict[str, np.typing.ArrayLike]:
        additional_observations = super().observations_hook()
        additional_observations.update({c.DEST: self[c.DEST].as_array()})
        return additional_observations

    def per_agent_reward_hook(self, agent: Agent) -> List[dict]:
        # noinspection PyUnresolvedReferences
        reward_event_list = super().per_agent_reward_hook(agent)
        if len(self[c.DEST_REACHED]):
            for reached_dest in list(self[c.DEST_REACHED]):
                if agent.pos == reached_dest.pos:
                    self.print(f'{agent.name} just reached destination at {agent.pos}')
                    self[c.DEST_REACHED].delete_env_object(reached_dest)
                    info_dict = {f'{agent.name}_{c.DEST_REACHED}': 1}
                    reward_event_list.append({'value': self.rewards_dest.DEST_REACHED,
                                              'reason': c.DEST_REACHED,
                                              'info': info_dict})
        return reward_event_list

    def render_assets_hook(self, mode='human'):
        # noinspection PyUnresolvedReferences
        additional_assets = super().render_assets_hook()
        destinations = [RenderEntity(c.DEST, dest.pos) for dest in self[c.DEST]]
        additional_assets.extend(destinations)
        return additional_assets


if __name__ == '__main__':
    from environments.utility_classes import AgentRenderOptions as aro, ObservationProperties

    render = True

    dest_probs = DestProperties(n_dests=2, spawn_frequency=5, spawn_mode=DestModeOptions.GROUPED)

    obs_props = ObservationProperties(render_agents=aro.LEVEL, omit_agent_self=True, pomdp_r=2)

    move_props = {'allow_square_movement': True,
                  'allow_diagonal_movement': False,
                  'allow_no_op': False}

    factory = DestFactory(n_agents=10, done_at_collision=False,
                          level_name='rooms', max_steps=400,
                          obs_prop=obs_props, parse_doors=True,
                          verbose=True,
                          mv_prop=move_props, dest_prop=dest_probs
                          )

    # noinspection DuplicatedCode
    n_actions = factory.action_space.n - 1
    _ = factory.observation_space

    for epoch in range(4):
        random_actions = [[random.randint(0, n_actions) for _
                           in range(factory.n_agents)] for _
                          in range(factory.max_steps + 1)]
        env_state = factory.reset()
        r = 0
        for agent_i_action in random_actions:
            env_state, step_r, done_bool, info_obj = factory.step(agent_i_action)
            r += step_r
            if render:
                factory.render()
            if done_bool:
                break
        print(f'Factory run {epoch} done, reward is:\n    {r}')
    pass
