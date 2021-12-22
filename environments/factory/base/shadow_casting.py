import numpy as np

from environments.helpers import Constants as c

mult_array = np.asarray([
    [1,  0,  0, -1, -1,  0,  0,  1],
    [0,  1, -1,  0,  0, -1,  1,  0],
    [0,  1,  1,  0,  0, -1, -1,  0],
    [1,  0,  0,  1, -1,  0,  0, -1]
])


class Map(object):
    # Multipliers for transforming coordinates to other octants:

    def __init__(self, map_array: np.typing.ArrayLike, diamond_slope: float = 0.9):
        self.data = map_array
        self.width, self.height = map_array.shape
        self.light = np.full_like(self.data, c.FREE_CELL.value)
        self.flag = c.FREE_CELL.value
        self.d_slope = diamond_slope

    def blocked(self, x, y):
        return (x < 0 or y < 0
                or x >= self.width or y >= self.height
                or self.data[x, y] == c.OCCUPIED_CELL.value)

    def lit(self, x, y):
        return self.light[x, y] == self.flag

    def set_lit(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.light[x, y] = self.flag

    def _cast_light(self, cx, cy, row, start, end, radius, xx, xy, yx, yy, id):
        "Recursive lightcasting function"
        if start < end:
            return
        radius_squared = radius*radius
        new_start = None
        for j in range(row, radius+1):
            dx, dy = -j-1, -j
            blocked = False
            while dx <= 0:
                dx += 1
                # Translate the dx, dy coordinates into map coordinates:
                X, Y = cx + dx * xx + dy * xy, cy + dx * yx + dy * yy
                # l_slope and r_slope store the slopes of the left and right
                # extremities of the square we're considering:
                l_slope, r_slope = (dx-self.d_slope)/(dy+self.d_slope), (dx+self.d_slope)/(dy-self.d_slope)
                if start < r_slope:
                    continue
                elif end > l_slope:
                    break
                else:
                    # Our light beam is touching this square; light it:
                    if dx*dx + dy*dy < radius_squared:
                        self.set_lit(X, Y)
                    if blocked:
                        # we're scanning a row of blocked squares:
                        if self.blocked(X, Y):
                            new_start = r_slope
                            continue
                        else:
                            blocked = False
                            start = new_start
                    else:
                        if self.blocked(X, Y) and j < radius:
                            # This is a blocking square, start a child scan:
                            blocked = True
                            self._cast_light(cx, cy, j+1, start, l_slope,
                                             radius, xx, xy, yx, yy, id+1)
                            new_start = r_slope
            # Row is scanned; do next row unless last square was blocked:
            if blocked:
                break

    def do_fov(self, x, y, radius):
        "Calculate lit squares from the given location and radius"
        self.flag += 1
        for oct in range(8):
            self._cast_light(x, y, 1, 1.0, 0.0, radius,
                             mult_array[0, oct], mult_array[1, oct],
                             mult_array[2, oct], mult_array[3, oct], 0)
        self.light[x, y] = self.flag
        return self.light
