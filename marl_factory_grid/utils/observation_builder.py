import math
from collections import defaultdict
from itertools import product
from typing import Dict, List

import numpy as np
from numba import njit

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.groups.utils import Combined
from marl_factory_grid.utils.states import Gamestate


class OBSBuilder(object):

    default_obs = [c.WALLS, c.OTHERS]

    @property
    def pomdp_d(self):
        if self.pomdp_r:
            return (self.pomdp_r * 2) + 1
        else:
            return 0

    def __init__(self, level_shape: np.size, state: Gamestate, pomdp_r: int):
        self.all_obs = dict()
        self.light_blockers = defaultdict(lambda: False)
        self.positional = defaultdict(lambda: False)
        self.non_positional = defaultdict(lambda: False)
        self.ray_caster = dict()

        self.level_shape = level_shape
        self.pomdp_r = pomdp_r
        self.obs_shape = (self.pomdp_d, self.pomdp_d) if self.pomdp_r else self.level_shape
        self.size = np.prod(self.obs_shape)

        self.obs_layers = dict()

        self.build_structured_obs_block(state)
        self.curr_lightmaps = dict()

    def build_structured_obs_block(self, state):
        self.all_obs[c.PLACEHOLDER] = np.full(self.obs_shape, 0, dtype=float)
        self.all_obs.update({key: obj for key, obj in state.entities.obs_pairs})

    def observation_space(self, state):
        from gymnasium.spaces import Tuple, Box
        obsn = self.refresh_and_build_for_all(state)
        if len(state[c.AGENT]) == 1:
            space = Box(low=0, high=1, shape=next(x for x in obsn.values()).shape, dtype=np.float32)
        else:
            space = Tuple([Box(low=0, high=1, shape=obs.shape, dtype=np.float32) for obs in obsn.values()])
        return space

    def named_observation_space(self, state):
        return self.refresh_and_build_for_all(state)

    def refresh_and_build_for_all(self, state) -> (dict, dict):
        self.build_structured_obs_block(state)
        info = {}
        return {agent.name: self.build_for_agent(agent, state)[0] for agent in state[c.AGENT]}, info

    def refresh_and_build_named_for_all(self, state) -> Dict[str, Dict[str, np.ndarray]]:
        self.build_structured_obs_block(state)
        named_obs_dict = {}
        for agent in state[c.AGENT]:
            obs, names = self.build_for_agent(agent, state)
            named_obs_dict[agent.name] = {'observation': obs, 'names': names}
        return named_obs_dict

    def build_for_agent(self, agent, state) -> (List[str], np.ndarray):
        try:
            agent_want_obs = self.obs_layers[agent.name]
        except KeyError:
            self._sort_and_name_observation_conf(agent)
            agent_want_obs = self.obs_layers[agent.name]

        # Handle in-grid observations aka visible observations
        visible_entitites = self.ray_caster[agent.name].visible_entities(state.entities)
        pre_sort_obs = defaultdict(lambda:  np.zeros((self.pomdp_d, self.pomdp_d)))
        for e in set(visible_entitites):
            x, y = (e.x - agent.x) + self.pomdp_r, (e.y - agent.y) + self.pomdp_r
            try:
                pre_sort_obs[e.obs_tag][x, y] += e.encoding
            except IndexError:
                # Seemded to be visible but is out or range
                pass

        pre_sort_obs = dict(pre_sort_obs)
        obs = np.zeros((len(agent_want_obs), self.pomdp_d, self.pomdp_d))

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
                            e = self.all_obs[f'{l_name}({agent.name})']
                        except KeyError:
                            try:
                                e = next(x for x in self.all_obs if l_name in x and agent.name in x)
                            except StopIteration:
                                raise KeyError(
                                    f'Check typing! {l_name} could not be found in: {list(dict(self.all_obs).keys())}')

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

        try:
            self.curr_lightmaps[agent.name] = pre_sort_obs[c.FLOORS].astype(bool)
        except KeyError:
            print()
        return obs, self.obs_layers[agent.name]

    def _sort_and_name_observation_conf(self, agent):
        self.ray_caster[agent.name] = RayCaster(agent, self.pomdp_r)
        obs_layers = []

        for obs_str in agent.observations:
            if isinstance(obs_str, dict):
                obs_str, vals = next(obs_str.items().__iter__())
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
                combined = Combined(names, self.pomdp_r, identifier=agent.name)
                self.all_obs[combined.name] = combined
                obs_layers.append(combined.name)
            elif obs_str == c.OTHERS:
                obs_layers.extend([x for x in self.all_obs if x != agent.name and x.startswith(f'{c.AGENT}[')])
            elif obs_str == c.AGENT:
                obs_layers.extend([x for x in self.all_obs if x.startswith(f'{c.AGENT}[')])
            else:
                obs_layers.append(obs_str)
        self.obs_layers[agent.name] = obs_layers
        self.curr_lightmaps[agent.name] = np.zeros((self.pomdp_d or self.level_shape[0],
                                                    self.pomdp_d or self.level_shape[1]
                                                    ))


class RayCaster:
    def __init__(self, agent, pomdp_r, degs=360):
        self.agent = agent
        self.pomdp_r = pomdp_r
        self.n_rays = 100  # (self.pomdp_r + 1) * 8
        self.degs = degs
        self.ray_targets = self.build_ray_targets()
        self.obs_shape_cube = np.array([self.pomdp_r, self.pomdp_r])

    def __repr__(self):
        return f'{self.__class__.__name__}({self.agent.name})'

    def build_ray_targets(self):
        north = np.array([0, -1])*self.pomdp_r
        thetas = [np.deg2rad(deg) for deg in np.linspace(-self.degs // 2, self.degs // 2, self.n_rays)[::-1]]
        rot_M = [
            [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]] for theta in thetas
        ]
        rot_M = np.stack(rot_M, 0)
        rot_M = np.unique(np.round(rot_M @ north), axis=0)
        return rot_M.astype(int)

    def ray_block_cache(self, cache_dict, key, callback):
        if key not in cache_dict:
            cache_dict[key] = callback()
        return cache_dict[key]

    def visible_entities(self, entities):
        visible = list()
        cache_blocking = {}

        for ray in self.get_rays():
            rx, ry = ray[0]
            for x, y in ray:
                cx, cy = x - rx, y - ry

                entities_hit = entities.pos_dict[(x, y)]
                hits = self.ray_block_cache(cache_blocking,
                                            (x, y),
                                            lambda: any(True for e in entities_hit if e.var_is_blocking_light))

                diag_hits = all([
                    self.ray_block_cache(
                        cache_blocking,
                        key,
                        lambda: all(False for e in entities.pos_dict[key] if not e.var_is_blocking_light))
                    for key in ((x, y-cy), (x-cx, y))
                ]) if (cx != 0 and cy != 0) else False

                visible += entities_hit if not diag_hits else []
                if hits or diag_hits:
                    break
                rx, ry = x, y
        return visible

    def get_rays(self):
        a_pos = self.agent.pos
        outline = self.ray_targets + a_pos
        return self.bresenham_loop(a_pos, outline)

    # todo do this once and cache the points!
    def get_fov_outline(self) -> np.ndarray:
        return self.ray_targets + self.agent.pos

    def get_square_outline(self):
        agent = self.agent
        x_coords = range(agent.x - self.pomdp_r, agent.x + self.pomdp_r + 1)
        y_coords = range(agent.y - self.pomdp_r, agent.y + self.pomdp_r + 1)
        outline = list(product(x_coords, [agent.y - self.pomdp_r, agent.y + self.pomdp_r])) \
                  + list(product([agent.x - self.pomdp_r, agent.x + self.pomdp_r], y_coords))
        return outline

    @staticmethod
    @njit
    def bresenham_loop(a_pos, points):
        results = []
        for end in points:
            x1, y1 = a_pos
            x2, y2 = end
            dx = x2 - x1
            dy = y2 - y1

            # Determine how steep the line is
            is_steep = abs(dy) > abs(dx)

            # Rotate line
            if is_steep:
                x1, y1 = y1, x1
                x2, y2 = y2, x2

            # Swap start and end points if necessary and store swap state
            swapped = False
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = True

            # Recalculate differentials
            dx = x2 - x1
            dy = y2 - y1

            # Calculate error
            error = int(dx / 2.0)
            ystep = 1 if y1 < y2 else -1

            # Iterate over bounding box generating points between start and end
            y = y1
            points = []
            for x in range(int(x1), int(x2) + 1):
                coord = [y, x] if is_steep else [x, y]
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += ystep
                    error += dx

            # Reverse the list if the coordinates were swapped
            if swapped:
                points.reverse()
            results.append(points)
        return results
