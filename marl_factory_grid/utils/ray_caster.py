import math
from itertools import product

import numpy as np
from numba import njit


class RayCaster:
    def __init__(self, agent, pomdp_r, degs=360):
        """
        TODO


        :return:
        """
        self.agent = agent
        self.pomdp_r = pomdp_r
        self.n_rays = 100  # (self.pomdp_r + 1) * 8
        self.degs = degs
        self.ray_targets = self.build_ray_targets()
        self.obs_shape_cube = np.array([self.pomdp_r, self.pomdp_r])
        self._cache_dict = {}

    def __repr__(self):
        return f'{self.__class__.__name__}({self.agent.name})'

    def build_ray_targets(self):
        north = np.array([0, -1]) * self.pomdp_r
        thetas = [np.deg2rad(deg) for deg in np.linspace(-self.degs // 2, self.degs // 2, self.n_rays)[::-1]]
        rot_M = [
            [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]] for theta in thetas
        ]
        rot_M = np.stack(rot_M, 0)
        rot_M = np.unique(np.round(rot_M @ north), axis=0)
        return rot_M.astype(int)

    def ray_block_cache(self, key, callback):
        if key not in self._cache_dict:
            self._cache_dict[key] = callback()
        return self._cache_dict[key]

    def visible_entities(self, pos_dict, reset_cache=True):
        visible = list()
        if reset_cache:
            self._cache_dict = dict()

        for ray in self.get_rays():  # Do not check, just trust.
            rx, ry = ray[0]
            # self.ray_block_cache(ray[0], lambda: False) We do not do that, because of doors etc...
            for x, y in ray:
                cx, cy = x - rx, y - ry

                entities_hit = pos_dict[(x, y)]
                hits = self.ray_block_cache((x, y),
                                            lambda: any(True for e in entities_hit if e.var_is_blocking_light)
                                            )

                diag_hits = all([
                    self.ray_block_cache(
                        key,
                        # lambda: all(False for e in pos_dict[key] if not e.var_is_blocking_light)
                        lambda: any(True for e in pos_dict[key] if e.var_is_blocking_light))
                    for key in ((x, y - cy), (x - cx, y))
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
        outline = list(product(x_coords, [agent.y - self.pomdp_r, agent.y + self.pomdp_r]))
        outline += list(product([agent.x - self.pomdp_r, agent.x + self.pomdp_r], y_coords))
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
