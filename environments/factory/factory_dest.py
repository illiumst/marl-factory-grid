import time
from collections import defaultdict
from enum import Enum
from typing import List, Union, NamedTuple, Dict
import numpy as np
import random

from environments.factory.base.base_factory import BaseFactory
from environments.helpers import Constants as c, Constants
from environments import helpers as h
from environments.factory.base.objects import Agent, Entity, Action
from environments.factory.base.registers import Entities, EntityRegister

from environments.factory.base.renderer import RenderEntity




DESTINATION = 1
DESTINATION_DONE = 0.5


class Destination(Entity):

    @property
    def any_agent_has_dwelled(self):
        return bool(len(self._per_agent_times))

    @property
    def currently_dwelling_names(self):
        return self._per_agent_times.keys()

    @property
    def can_collide(self):
        return False

    @property
    def encoding(self):
        return DESTINATION

    def __init__(self, *args, dwell_time: int = 0, **kwargs):
        super(Destination, self).__init__(*args, **kwargs)
        self.dwell_time = dwell_time
        self._per_agent_times = defaultdict(lambda: dwell_time)

    def wait(self, agent: Agent):
        self._per_agent_times[agent.name] -= 1
        return c.VALID

    def leave(self, agent: Agent):
        del self._per_agent_times[agent.name]

    @property
    def is_considered_reached(self):
        agent_at_position = any(c.AGENT.name.lower() in x.name.lower() for x in self.tile.guests_that_can_collide)
        return (agent_at_position and not self.dwell_time) or any(x == 0 for x in self._per_agent_times.values())

    def agent_is_dwelling(self, agent: Agent):
        return self._per_agent_times[agent.name] < self.dwell_time

    def summarize_state(self, n_steps=None) -> dict:
        state_summary = super().summarize_state(n_steps=n_steps)
        state_summary.update(per_agent_times=self._per_agent_times)
        return state_summary


class Destinations(EntityRegister):

    _accepted_objects = Destination
    _light_blocking = False

    def as_array(self):
        self._array[:] = c.FREE_CELL.value
        # ToDo: Switch to new Style Array Put
        # indices = list(zip(range(len(cls)), *zip(*[x.pos for x in cls])))
        # np.put(cls._array, [np.ravel_multi_index(x, cls._array.shape) for x in indices], cls.encodings)
        for item in self:
            if item.pos != c.NO_POS.value:
                self._array[0, item.x, item.y] = item.encoding
        return self._array

    def __repr__(self):
        super(Destinations, self).__repr__()


class ReachedDestinations(Destinations):
    _accepted_objects = Destination
    _light_blocking = False

    def __init__(self, *args, **kwargs):
        super(ReachedDestinations, self).__init__(*args, **kwargs)

    def summarize_states(self, n_steps=None):
        return {}


class DestModeOptions(object):
    DONE        = 'DONE'
    GROUPED     = 'GROUPED'
    PER_DEST    = 'PER_DEST'


class DestProperties(NamedTuple):
    n_dests:                                     int = 1     # How many destinations are there
    dwell_time:                                  int = 0     # How long does the agent need to "wait" on a destination
    spawn_frequency:                             int = 0
    spawn_in_other_zone:                        bool = True  #
    spawn_mode:                                  str = DestModeOptions.DONE

    assert dwell_time >= 0, 'dwell_time cannot be < 0!'
    assert spawn_frequency >= 0, 'spawn_frequency cannot be < 0!'
    assert n_dests >= 0, 'n_destinations cannot be < 0!'
    assert (spawn_mode == DestModeOptions.DONE) != bool(spawn_frequency)


# noinspection PyAttributeOutsideInit, PyAbstractClass
class DestFactory(BaseFactory):
    # noinspection PyMissingConstructor

    def __init__(self, *args, dest_prop: DestProperties  = DestProperties(),
                 env_seed=time.time_ns(), **kwargs):
        if isinstance(dest_prop, dict):
            dest_prop = DestProperties(**dest_prop)
        self.dest_prop = dest_prop
        kwargs.update(env_seed=env_seed)
        self._dest_rng = np.random.default_rng(env_seed)
        super().__init__(*args, **kwargs)

    @property
    def additional_actions(self) -> Union[Action, List[Action]]:
        # noinspection PyUnresolvedReferences
        super_actions = super().additional_actions
        if self.dest_prop.dwell_time:
            super_actions.append(Action(enum_ident=h.EnvActions.WAIT_ON_DEST))
        return super_actions

    @property
    def additional_entities(self) -> Dict[(Enum, Entities)]:
        # noinspection PyUnresolvedReferences
        super_entities = super().additional_entities

        empty_tiles = self[c.FLOOR].empty_tiles[:self.dest_prop.n_dests]
        destinations = Destinations.from_tiles(
            empty_tiles, self._level_shape,
            entity_kwargs=dict(
                dwell_time=self.dest_prop.dwell_time)
        )
        reached_destinations = ReachedDestinations(level_shape=self._level_shape)

        super_entities.update({c.DESTINATION: destinations, c.REACHEDDESTINATION: reached_destinations})
        return super_entities

    def wait(self, agent: Agent):
        if destiantion := self[c.DESTINATION].by_pos(agent.pos):
            valid = destiantion.wait(agent)
            return valid
        else:
            return c.NOT_VALID

    def do_additional_actions(self, agent: Agent, action: Action) -> Union[None, c]:
        # noinspection PyUnresolvedReferences
        valid = super().do_additional_actions(agent, action)
        if valid is None:
            if action == h.EnvActions.WAIT_ON_DEST:
                valid = self.wait(agent)
                return valid
            else:
                return None
        else:
            return valid

    def do_additional_reset(self) -> None:
        # noinspection PyUnresolvedReferences
        super().do_additional_reset()
        self._dest_spawn_timer = dict()

    def trigger_destination_spawn(self):
        destinations_to_spawn = [key for key, val in self._dest_spawn_timer.items()
                                 if val == self.dest_prop.spawn_frequency]
        if destinations_to_spawn:
            n_dest_to_spawn = len(destinations_to_spawn)
            if self.dest_prop.spawn_mode != DestModeOptions.GROUPED:
                destinations = [Destination(tile) for tile in self[c.FLOOR].empty_tiles[:n_dest_to_spawn]]
                self[c.DESTINATION].register_additional_items(destinations)
                for dest in destinations_to_spawn:
                    del self._dest_spawn_timer[dest]
                self.print(f'{n_dest_to_spawn} new destinations have been spawned')
            elif self.dest_prop.spawn_mode == DestModeOptions.GROUPED and n_dest_to_spawn == self.dest_prop.n_dests:
                destinations = [Destination(tile) for tile in self[c.FLOOR].empty_tiles[:n_dest_to_spawn]]
                self[c.DESTINATION].register_additional_items(destinations)
                for dest in destinations_to_spawn:
                    del self._dest_spawn_timer[dest]
                self.print(f'{n_dest_to_spawn} new destinations have been spawned')
            else:
                self.print(f'{n_dest_to_spawn} new destinations could be spawned, but waiting for all.')
                pass
        else:
            self.print('No Items are spawning, limit is reached.')

    def do_additional_step(self) -> dict:
        # noinspection PyUnresolvedReferences
        info_dict = super().do_additional_step()
        for key, val in self._dest_spawn_timer.items():
            self._dest_spawn_timer[key] = min(self.dest_prop.spawn_frequency, self._dest_spawn_timer[key] + 1)
        for dest in list(self[c.DESTINATION].values()):
            if dest.is_considered_reached:
                self[c.REACHEDDESTINATION].register_item(dest)
                self[c.DESTINATION].delete_env_object(dest)
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
        return info_dict

    def _additional_observations(self) -> Dict[Constants, np.typing.ArrayLike]:
        additional_observations = super()._additional_observations()
        additional_observations.update({c.DESTINATION: self[c.DESTINATION].as_array()})
        return additional_observations

    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        # noinspection PyUnresolvedReferences
        reward, info_dict = super().calculate_additional_reward(agent)
        if h.EnvActions.WAIT_ON_DEST == agent.temp_action:
            if agent.temp_valid:
                info_dict.update({f'{agent.name}_waiting_at_dest': 1})
                info_dict.update(agent_waiting_at_dest=1)
                self.print(f'{agent.name} just waited at {agent.pos}')
                reward += 0.1
            else:
                info_dict.update({f'{agent.name}_tried_failed': 1})
                info_dict.update(agent_waiting_failed=1)
                self.print(f'{agent.name} just tried to wait wait at {agent.pos} but failed')
                reward -= 0.1
        if len(self[c.REACHEDDESTINATION]):
            for reached_dest in list(self[c.REACHEDDESTINATION]):
                if agent.pos == reached_dest.pos:
                    info_dict.update({f'{agent.name}_reached_destination': 1})
                    info_dict.update(agent_reached_destination=1)
                    self.print(f'{agent.name} just reached destination at {agent.pos}')
                    reward += 0.5
                    self[c.REACHEDDESTINATION].delete_env_object(reached_dest)
        return reward, info_dict

    def render_additional_assets(self, mode='human'):
        # noinspection PyUnresolvedReferences
        additional_assets = super().render_additional_assets()
        destinations = [RenderEntity(c.DESTINATION.value, dest.pos) for dest in self[c.DESTINATION]]
        additional_assets.extend(destinations)
        return additional_assets


if __name__ == '__main__':
    from environments.utility_classes import AgentRenderOptions as ARO, ObservationProperties

    render = True

    dest_probs = DestProperties(n_dests=2, spawn_frequency=5, spawn_mode=DestModeOptions.GROUPED)

    obs_props = ObservationProperties(render_agents=ARO.LEVEL, omit_agent_self=True, pomdp_r=2)

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
