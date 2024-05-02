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

    def __init__(self, lvl_shape: Tuple[int, int] = (16, 16), lvl_padded_shape: Union[Tuple[int, int], None] = None,
                 cell_size: int = 40, fps: int = 7, factor: float = 0.9, grid_lines: bool = True, view_radius: int = 2,
                 custom_assets_path=None):
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
        self.custom_assets_path = custom_assets_path
        self.assets = self.load_assets(custom_assets_path)
        self.save_counter = 1
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

    def load_assets(self, custom_assets_path):
        """
        Loads assets from the custom path if provided, otherwise from the default path.
        """
        assets_directory = custom_assets_path if custom_assets_path else self.ASSETS
        assets = {}
        if isinstance(assets_directory, dict):
            for key, path in assets_directory.items():
                asset = self.load_asset(path)
                if asset is not None:
                    assets[key] = asset
                else:
                    print(f"Warning: Asset for key '{key}' is missing and was not loaded.")
        else:
            for path in Path(assets_directory).rglob('*.png'):
                asset = self.load_asset(str(path))
                if asset is not None:
                    assets[path.stem] = asset
                else:
                    print(f"Warning: Asset '{path.stem}' is missing and was not loaded.")
        return assets

    def load_asset(self, path, factor=1.0):
        """
        Loads and resizes an asset from the specified path.

        :param path: Path to the asset.
        :type path: str
        :param factor: Resizing factor for the asset.
        :type factor: float
        :return: Resized asset.
        """
        try:
            s = int(factor * self.cell_size)
            asset = pygame.image.load(path).convert_alpha()
            asset = pygame.transform.smoothscale(asset, (s, s))
            return asset
        except pygame.error as e:
            print(f"Failed to load asset {path}: {e}")
            return self.load_default_asset()

    def load_default_asset(self, factor=1.0):
        """
        Loads a default asset to be used when specific assets fail to load.
        """
        default_path = 'marl_factory_grid/utils/plotting/action_assets/default.png'
        try:
            s = int(factor * self.cell_size)
            default_asset = pygame.image.load(default_path).convert_alpha()
            default_asset = pygame.transform.smoothscale(default_asset, (s, s))
            return default_asset
        except pygame.error as e:
            print(f"Failed to load default asset: {e}")
            return None

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

    def render(self, entities, recorder):
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

        if recorder:
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))  # Transpose to (height, width, channels)
            recorder.append_data(frame)

        pygame.display.flip()
        self.clock.tick(self.fps)
        rgb_obs = pygame.surfarray.array3d(self.screen)
        return np.transpose(rgb_obs, (2, 0, 1))
        # return torch.from_numpy(rgb_obs).permute(2, 0, 1)

    def render_action_icons(self, action_entities):
        """
        Renders action icons based on the entities' specified actions, positions, and probabilities.

        :param action_entities: List of entities representing actions.
        :type action_entities: List[RenderEntity]
        """

        self.fill_bg()  # Clear the background
        font = pygame.font.Font(None, 24)  # Initialize the font once for all text rendering

        for action_entity in action_entities:
            if not isinstance(action_entity.pos, np.ndarray) or action_entity.pos.ndim != 1:
                print(f"Invalid position format for entity: {action_entity.pos}")
                continue

            # Load and potentially rotate the icon based on action name
            img = self.assets[action_entity.name.lower()]
            if hasattr(action_entity, 'rotation'):
                img = pygame.transform.rotate(img, action_entity.rotation)
                if img is None:
                    print(f"Error: No asset available for '{action_entity.name}'. Skipping rendering this entity.")
                    continue

            # Blit the icon image
            img_rect = img.get_rect(center=(action_entity.pos[0] * self.cell_size + self.cell_size // 2,
                                            action_entity.pos[1] * self.cell_size + self.cell_size // 2))
            self.screen.blit(img, img_rect)

            # Render the probability next to the icon if it exists
            if hasattr(action_entity, 'probability'):
                prob_text = font.render(f"{action_entity.probability:.2f}", True, (255, 0, 0))
                prob_text_rect = prob_text.get_rect(top=img_rect.bottom, left=img_rect.left)
                self.screen.blit(prob_text, prob_text_rect)

        pygame.display.flip()  # Update the display with all new blits
        self.save_screen("route_graph")

    def save_screen(self, filename):
        """
        Saves the current screen to a PNG file, appending a counter to ensure uniqueness.
        :param filename: The base filename where to save the image.
        :param agent_id: Unique identifier for the agent.
        """
        unique_filename = f"{filename}_agent_{self.save_counter}.png"
        self.save_counter += 1
        pygame.image.save(self.screen, unique_filename)
        print(f"Image saved as {unique_filename}")


if __name__ == '__main__':
    renderer = Renderer(cell_size=40, fps=2)
    for pos_i in range(15):
        entity_1 = RenderEntity('agent_collision', [5, pos_i], 1, 'idle', 'idle')
        renderer.render([entity_1])
