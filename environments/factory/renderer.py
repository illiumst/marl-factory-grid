import sys
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from collections import deque
import pygame


@dataclass
class Entity:
    name: str
    pos: np.array
    value: float = 1
    value_operation: str = 'none'


class Renderer:
    BG_COLOR = (178, 190, 195)#(99, 110, 114)
    WHITE = (223, 230, 233)#(200, 200, 200)
    AGENT_VIEW_COLOR = (9, 132, 227)

    def __init__(self, grid_w=16, grid_h=16, cell_size=40, fps=4,  grid_lines=True, view_radius=2):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.cell_size = cell_size
        self.fps = fps
        self.grid_lines = grid_lines
        self.view_radius = view_radius
        pygame.init()
        self.screen_size = (grid_h*cell_size, grid_w*cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        assets = list((Path(__file__).parent / 'assets').rglob('*.png'))
        self.assets = {path.stem: self.load_asset(str(path), 1) for path in assets}
        self.fill_bg()

    def fill_bg(self):
        self.screen.fill(Renderer.BG_COLOR)
        if self.grid_lines:
            h, w = self.screen_size
            for x in range(0, w, self.cell_size):
                for y in range(0, h, self.cell_size):
                    rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, Renderer.WHITE, rect, 1)

    def blit_params(self, entity):
        r, c = entity.pos
        img = self.assets[entity.name]
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
        wall_img = pygame.image.load(path).convert_alpha()
        wall_img = pygame.transform.smoothscale(wall_img, (s, s))
        return wall_img

    def render(self, pos_dict):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.fill_bg()
        blits = deque()
        for asset, entities in pos_dict.items():
            for entity in entities:
                bp = self.blit_params(entity)
                if 'agent' in entity.name and self.view_radius > 0:
                    visibility_rect = bp['dest'].inflate((self.view_radius*2)*self.cell_size, (self.view_radius*2)*self.cell_size)
                    shape_surf = pygame.Surface(visibility_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(shape_surf, self.AGENT_VIEW_COLOR, shape_surf.get_rect())
                    shape_surf.set_alpha(64)
                    blits.appendleft(dict(source=shape_surf, dest=visibility_rect))
                blits.append(bp)
        for blit in blits: self.screen.blit(**blit)
        pygame.display.flip()
        self.clock.tick(self.fps)


if __name__ == '__main__':
    renderer = Renderer(fps=2, cell_size=40)
    for i in range(15):
        renderer.render({'agent': [(5, i)], 'wall': [(0, i), (i, 0)], 'dirt': [(3,3), (3,4)]})

