import sys

from pathlib import Path
from collections import deque
from itertools import product

import numpy as np
import pygame
from typing import Tuple, Union
import time

from marl_factory_grid.utils.utility_classes import RenderEntity

AGENT: str = 'agent'
STATE_IDLE: str = 'idle'
STATE_MOVE: str = 'move'
STATE_VALID: str = 'valid'
STATE_INVALID: str = 'invalid'
STATE_COLLISION: str = 'agent_collision'
BLANK: str = 'blank'
DOOR: str = 'door'
OPACITY: str = 'opacity'
SCALE: str = 'scale'


class Renderer:
    BG_COLOR = (178, 190, 195)         # (99, 110, 114)
    WHITE = (223, 230, 233)            # (200, 200, 200)
    AGENT_VIEW_COLOR = (9, 132, 227)
    ASSETS = Path(__file__).parent.parent

    def __init__(self, lvl_shape: Tuple[int, int] = (16, 16),
                 lvl_padded_shape: Union[Tuple[int, int], None] = None,
                 cell_size: int = 40, fps: int = 7, factor: float = 0.9,
                 grid_lines: bool = True, view_radius: int = 2):
        """
        The Renderer class initializes and manages the rendering environment for the simulation,
        providing methods for preparing entities for display, loading assets, calculating visibility rectangles and
        rendering the entities on the screen with specified parameters.

        :param lvl_shape: Tuple representing the shape of the level.
        :type lvl_shape: Tuple[int, int]
        :param lvl_padded_shape: Optional Tuple representing the padded shape of the level.
        :type lvl_padded_shape: Union[Tuple[int, int], None]
        :param cell_size: Size of each cell in pixels.
        :type cell_size: int
        :param fps: Frames per second for rendering.
        :type fps: int
        :param factor: Factor for resizing assets.
        :type factor: float
        :param grid_lines: Boolean indicating whether to display grid lines.
        :type grid_lines: bool
        :param view_radius: Radius for agent's field of view.
        :type view_radius: int
        """
        # TODO: Custom_assets paths
        self.grid_h, self.grid_w = lvl_shape
        self.lvl_padded_shape = lvl_padded_shape if lvl_padded_shape is not None else lvl_shape
        self.cell_size = cell_size
        self.fps = fps
        self.grid_lines = grid_lines
        self.view_radius = view_radius
        pygame.init()
        self.screen_size = (self.grid_w*cell_size, self.grid_h*cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        assets = list(self.ASSETS.rglob('*.png'))
        self.assets = {path.stem: self.load_asset(str(path), factor) for path in assets}
        self.fill_bg()

        now = time.time()
        self.font = pygame.font.Font(None, 20)
        self.font.set_bold(True)
        print('Loading System font with pygame.font.Font took', time.time() - now)

    def fill_bg(self):
        """
        Fills the background of the screen with the specified BG color.
        """
        self.screen.fill(Renderer.BG_COLOR)
        if self.grid_lines:
            w, h = self.screen_size
            for x in range(0, w, self.cell_size):
                for y in range(0, h, self.cell_size):
                    rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, Renderer.WHITE, rect, 1)

    def blit_params(self, entity):
        """
        Prepares parameters for blitting an entity on the screen. Blitting refers to the process of combining or copying
        rectangular blocks of pixels from one part of a graphical buffer to another and is often used to efficiently
        update the display by copying pre-drawn or cached images onto the screen.

        :param entity: The entity to be blitted.
        :type entity: Entity
        :return: Dictionary containing source and destination information for blitting.
        :rtype: dict
        """
        offset_r, offset_c = (self.lvl_padded_shape[0] - self.grid_h) // 2, \
                             (self.lvl_padded_shape[1] - self.grid_w) // 2

        r, c = entity.pos
        r, c = r - offset_r, c-offset_c

        img = self.assets[entity.name.lower()]
        if entity.value_operation == OPACITY:
            img.set_alpha(255*entity.value)
        elif entity.value_operation == SCALE:
            re = img.get_rect()
            img = pygame.transform.smoothscale(
                img, (int(entity.value*re.width), int(entity.value*re.height))
            )
        o = self.cell_size//2
        r_, c_ = r*self.cell_size + o, c*self.cell_size + o
        rect = img.get_rect()
        rect.centerx, rect.centery = c_, r_
        return dict(source=img, dest=rect)

    def load_asset(self, path, factor=1.0):
        """
        Loads and resizes an asset from the specified path.

        :param path: Path to the asset.
        :type path: str
        :param factor: Resizing factor for the asset.
        :type factor: float
        :return: Resized asset.
        """
        s = int(factor*self.cell_size)
        asset = pygame.image.load(path).convert_alpha()
        asset = pygame.transform.smoothscale(asset, (s, s))
        return asset

    def visibility_rects(self, bp, view):
        """
        Calculates the visibility rectangles for an agent.

        :param bp: Blit parameters for the agent.
        :type bp: dict
        :param view: Agent's field of view.
        :type view: np.ndarray
        :return: List of visibility rectangles.
        :rtype: List[dict]
        """
        rects = []
        for i, j in product(range(-self.view_radius, self.view_radius+1),
                            range(-self.view_radius, self.view_radius+1)):
            if view is not None:
                if bool(view[self.view_radius+j, self.view_radius+i]):
                    visibility_rect = bp['dest'].copy()
                    visibility_rect.centerx += i*self.cell_size
                    visibility_rect.centery += j*self.cell_size
                    shape_surf = pygame.Surface(visibility_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(shape_surf, self.AGENT_VIEW_COLOR, shape_surf.get_rect())
                    shape_surf.set_alpha(64)
                    rects.append(dict(source=shape_surf, dest=visibility_rect))
        return rects

    def render(self, entities):
        """
        Renders the entities on the screen.

        :param entities: List of entities to be rendered.
        :type entities: List[Entity]
        :return: Transposed RGB observation array.
        :rtype: np.ndarray
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.fill_bg()
        # First all others
        blits = deque(self.blit_params(x) for x in entities if not x.name.lower() == AGENT)
        # Then Agents, so that agents are rendered on top.
        for agent in (x for x in entities if x.name.lower() == AGENT):
            agent_blit = self.blit_params(agent)
            if self.view_radius > 0:
                vis_rects = self.visibility_rects(agent_blit, agent.aux)
                blits.extendleft(vis_rects)
            if agent.state != BLANK:
                state_blit = self.blit_params(
                    RenderEntity(agent.state, (agent.pos[0] + 0.12, agent.pos[1]), 0.48, SCALE)
                )
                textsurface = self.font.render(str(agent.id), False, (0, 0, 0))
                text_blit = dict(source=textsurface, dest=(agent_blit['dest'].center[0]-.07*self.cell_size,
                                                           agent_blit['dest'].center[1]))
                blits += [agent_blit, state_blit, text_blit]

        for blit in blits:
            self.screen.blit(**blit)

        pygame.display.flip()
        self.clock.tick(self.fps)
        rgb_obs = pygame.surfarray.array3d(self.screen)
        return np.transpose(rgb_obs, (2, 0, 1))
        # return torch.from_numpy(rgb_obs).permute(2, 0, 1)


if __name__ == '__main__':
    renderer = Renderer(fps=2, cell_size=40)
    for pos_i in range(15):
        entity_1 = RenderEntity('agent_collision', [5, pos_i], 1, 'idle', 'idle')
        renderer.render([entity_1])
