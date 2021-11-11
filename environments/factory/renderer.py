import sys
import numpy as np
from pathlib import Path
from collections import deque
from itertools import product
import pygame
from typing import NamedTuple, Any
import time


class RenderEntity(NamedTuple):
    name: str
    pos: np.array
    value: float = 1
    value_operation: str = 'none'
    state: str = None
    id: int = 0
    aux: Any = None


class Renderer:
    BG_COLOR = (178, 190, 195)         # (99, 110, 114)
    WHITE = (223, 230, 233)            # (200, 200, 200)
    AGENT_VIEW_COLOR = (9, 132, 227)
    ASSETS = Path(__file__).parent / 'assets'

    def __init__(self, grid_w=16, grid_h=16, cell_size=40, fps=7,  grid_lines=True, view_radius=2):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.cell_size = cell_size
        self.fps = fps
        self.grid_lines = grid_lines
        self.view_radius = view_radius
        pygame.init()
        self.screen_size = (grid_w*cell_size, grid_h*cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        assets = list(self.ASSETS.rglob('*.png'))
        self.assets = {path.stem: self.load_asset(str(path), 1) for path in assets}
        self.fill_bg()

        now = time.time()
        self.font = pygame.font.Font(None, 20)
        self.font.set_bold(1)
        print('Loading System font with pygame.font.Font took', time.time() - now)

    def fill_bg(self):
        self.screen.fill(Renderer.BG_COLOR)
        if self.grid_lines:
            w, h = self.screen_size
            for x in range(0, w, self.cell_size):
                for y in range(0, h, self.cell_size):
                    rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, Renderer.WHITE, rect, 1)

    def blit_params(self, entity):
        r, c = entity.pos
        img = self.assets[entity.name.lower()]
        if entity.value_operation == 'opacity':
            img.set_alpha(255*entity.value)
        elif entity.value_operation == 'scale':
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
        s = int(factor*self.cell_size)
        asset = pygame.image.load(path).convert_alpha()
        asset = pygame.transform.smoothscale(asset, (s, s))
        return asset

    def visibility_rects(self, bp, view):
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.fill_bg()
        blits = deque()
        for entity in [x for x in entities if 'door' in x.name]:
            bp = self.blit_params(entity)
            blits.append(bp)
        for entity in [x for x in entities if 'door' not in x.name]:
            bp = self.blit_params(entity)
            blits.append(bp)
            if entity.name.lower() == 'agent':
                if self.view_radius > 0:
                    vis_rects = self.visibility_rects(bp, entity.aux)
                    blits.extendleft(vis_rects)
                if entity.state != 'blank':
                    agent_state_blits = self.blit_params(
                        RenderEntity(entity.state, (entity.pos[0] + 0.12, entity.pos[1]), 0.48, 'scale')
                    )
                    textsurface = self.font.render(str(entity.id), False, (0, 0, 0))
                    text_blit = dict(source=textsurface, dest=(bp['dest'].center[0]-.07*self.cell_size,
                                                               bp['dest'].center[1]))
                    blits += [agent_state_blits, text_blit]

        for blit in blits:
            self.screen.blit(**blit)

        pygame.display.flip()
        self.clock.tick(self.fps)


if __name__ == '__main__':
    renderer = Renderer(fps=2, cell_size=40)
    for i in range(15):
        entity_1 = RenderEntity('agent_collision', [5, i], 1, 'idle', 'idle')
        renderer.render([entity_1])

