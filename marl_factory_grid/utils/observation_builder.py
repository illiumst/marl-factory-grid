import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.object import Object
from marl_factory_grid.environment.groups.utils import Combined
from marl_factory_grid.utils.utility_classes import Floor
from marl_factory_grid.utils.ray_caster import RayCaster
from marl_factory_grid.utils.states import Gamestate
from marl_factory_grid.utils import helpers as h


class OBSBuilder(object):
    default_obs = [c.WALLS, c.OTHERS]

    @property
    def pomdp_d(self):
        """
        TODO


        :return:
        """
        if self.pomdp_r:
            return (self.pomdp_r * 2) + 1
        else:
            return 0

    def __init__(self, level_shape: np.size, state: Gamestate, pomdp_r: int):
        """
        TODO


        :return:
        """
        self.all_obs = dict()
        self.ray_caster = dict()

        self.level_shape = level_shape
        self.pomdp_r = pomdp_r
        self.obs_shape = (self.pomdp_d, self.pomdp_d) if self.pomdp_r else self.level_shape
        self.size = np.prod(self.obs_shape)

        self.obs_layers = dict()
        self.curr_lightmaps = dict()

        self._floortiles = defaultdict(list, {pos: [Floor(*pos)] for pos in state.entities.floorlist})

        self.reset(state)

    def reset(self, state):
        # Reset temporary information
        self.curr_lightmaps = dict()
        # Construct an empty obs (array) for possible placeholders
        self.all_obs[c.PLACEHOLDER] = np.full(self.obs_shape, 0, dtype=float)
        # Fill the all_obs-dict with all available entities
        self.all_obs.update({key: obj for key, obj in state.entities.obs_pairs})
        return True

    def observation_space(self, state):
        from gymnasium.spaces import Tuple, Box
        self.reset(state)
        obsn = self.build_for_all(state)
        if len(state[c.AGENT]) == 1:
            space = Box(low=0, high=1, shape=next(x for x in obsn.values()).shape, dtype=np.float32)
        else:
            space = Tuple([Box(low=0, high=1, shape=obs.shape, dtype=np.float32) for obs in obsn.values()])
        return space

    def named_observation_space(self, state):
        self.reset(state)
        return self.build_for_all(state)

    def build_for_all(self, state) -> (dict, dict):
        return {agent.name: self.build_for_agent(agent, state)[0] for agent in state[c.AGENT]}

    def build_named_for_all(self, state) -> Dict[str, Dict[str, np.ndarray]]:
        named_obs_dict = {}
        for agent in state[c.AGENT]:
            obs, names = self.build_for_agent(agent, state)
            named_obs_dict[agent.name] = {'observation': obs, 'names': names}
        return named_obs_dict

    def place_entity_in_observation(self, obs_array, agent, e):
        x, y = (e.x - agent.x) + self.pomdp_r, (e.y - agent.y) + self.pomdp_r
        if not min([y, x]) < 0:
            try:
                obs_array[x, y] += e.encoding
            except IndexError:
                # Seemded to be visible but is out of range
                pass
        pass

    def build_for_agent(self, agent, state) -> (List[str], np.ndarray):
        try:
            agent_want_obs = self.obs_layers[agent.name]
        except KeyError:
            self._sort_and_name_observation_conf(agent)
            agent_want_obs = self.obs_layers[agent.name]

        # Handle in-grid observations aka visible observations (Things on the map, with pos)
        visible_entities = self.ray_caster[agent.name].visible_entities(state.entities.pos_dict)
        pre_sort_obs = defaultdict(lambda: np.zeros(self.obs_shape))
        if self.pomdp_r:
            for e in set(visible_entities):
                self.place_entity_in_observation(pre_sort_obs[e.obs_tag], agent, e)
        else:
            for e in set(visible_entities):
                pre_sort_obs[e.obs_tag][e.x, e.y] += e.encoding

        pre_sort_obs = dict(pre_sort_obs)
        obs = np.zeros((len(agent_want_obs), self.obs_shape[0], self.obs_shape[1]))

        for idx, l_name in enumerate(agent_want_obs):
            try:
                obs[idx] = pre_sort_obs[l_name]
            except KeyError:
                if c.COMBINED in l_name:
                    if combined := [pre_sort_obs[x] for x in self.all_obs[f'{c.COMBINED}({agent.name})'].names
                                    if x in pre_sort_obs]:
                        obs[idx] = np.sum(combined, axis=0)
                elif l_name == c.PLACEHOLDER:
                    obs[idx] = self.all_obs[c.PLACEHOLDER]
                else:
                    try:
                        e = self.all_obs[l_name]
                    except KeyError:
                        try:
                            # Look for bound entity REPRs!
                            pattern = re.compile(f'{re.escape(l_name)}'
                                                 f'{re.escape("[")}(.*){re.escape("]")}'
                                                 f'{re.escape("(")}{re.escape(agent.name)}{re.escape(")")}')
                            name = next((key for key, val in self.all_obs.items()
                                         if pattern.search(str(val)) and isinstance(val, Object)), None)
                            e = self.all_obs[name]
                        except KeyError:
                            try:
                                e = next(v for k, v in self.all_obs.items() if l_name in k and agent.name in k)
                            except StopIteration:
                                print(f'# Check for spelling errors!')
                                print(f'# No combination of "{l_name}" and "{agent.name}" could not be found in:')
                                print(f'# {list(dict(self.all_obs).keys())}')
                                print('#')
                                print('# exiting...')
                                print('#')
                                exit(-99999)

                    try:
                        positional = e.var_has_position
                    except AttributeError:
                        positional = False
                    if positional:
                        # Seems to be not visible, so just skip it
                        # obs[idx] = np.zeros((self.pomdp_d, self.pomdp_d))
                        # All good
                        pass
                    else:
                        try:
                            v = e.encodings
                        except AttributeError:
                            try:
                                v = e.encoding
                            except AttributeError:
                                raise AttributeError(f'This env. expects Entity-Clases to report their "encoding"')
                        try:
                            np.put(obs[idx], range(len(v)), v, mode='raise')
                        except TypeError:
                            np.put(obs[idx], 0, v, mode='raise')
                        except IndexError:
                            raise ValueError(f'Max(obs.size) for {e.name}:  {obs[idx].size}, but was: {len(v)}.')
        if self.pomdp_r:
            try:
                light_map = self.curr_lightmaps.get(agent.name, np.zeros(self.obs_shape))
                light_map[:] = 0.0
                visible_floor = self.ray_caster[agent.name].visible_entities(self._floortiles, reset_cache=False)

                for f in set(visible_floor):
                    self.place_entity_in_observation(light_map, agent, f)
                # else:
                #     for f in set(visible_floor):
                #         light_map[f.x, f.y] += f.encoding
                self.curr_lightmaps[agent.name] = light_map
            except (KeyError, ValueError):
                pass
        return obs, self.obs_layers[agent.name]

    def _sort_and_name_observation_conf(self, agent):
        """
        Builds the useable observation scheme per agent from conf.yaml.
        :param agent:
        :return:
        """
        # Fixme: no asymetric shapes possible.
        self.ray_caster[agent.name] = RayCaster(agent, min(self.obs_shape))
        obs_layers = []

        for obs_str in agent.observations:
            if isinstance(obs_str, dict):
                obs_str, vals = h.get_first(obs_str.items())
            else:
                vals = None
            if obs_str == c.SELF:
                obs_layers.append(agent.name)
            elif obs_str == c.DEFAULTS:
                obs_layers.extend(self.default_obs)
            elif obs_str == c.COMBINED:
                if isinstance(vals, str):
                    vals = [vals]
                names = list()
                for val in vals:
                    if val == c.SELF:
                        names.append(agent.name)
                    elif val == c.OTHERS:
                        names.extend([x.name for x in agent.collection if x.name != agent.name])
                    else:
                        names.append(val)
                combined = Combined(names, self.size, identifier=agent.name)
                self.all_obs[combined.name] = combined
                obs_layers.append(combined.name)
            elif obs_str == c.OTHERS:
                obs_layers.extend([x for x in self.all_obs if x != agent.name and x.startswith(f'{c.AGENT}[')])
            elif obs_str == c.AGENT:
                obs_layers.extend([x for x in self.all_obs if x.startswith(f'{c.AGENT}[')])
            else:
                obs_layers.append(obs_str)
        self.obs_layers[agent.name] = obs_layers
        self.curr_lightmaps[agent.name] = np.zeros(self.obs_shape)
