from typing import Dict

import pygame
from pygame import Rect

from src.ui import colors
from src.ui.input_handler import InputHandler
from src.ui.text_renderer import TextRenderer


class Joint:
    def __init__(self, coords: tuple[float, float], pos: tuple[float, float], id: str):
        self.id = id
        self.coords = coords
        self.pos = pos
        self.selected = False
        self.rect = Rect(self.coords[0] - 21, self.coords[1] - 21, 42, 42)

        self.bones = []

    def show(self, window):
        if self.selected:
            pygame.draw.rect(window, colors.foreground, self.rect, border_radius=100)
        pygame.draw.circle(window, colors.background_secondary, self.coords, 20)
        TextRenderer.render_text(self.id, 13, colors.foreground, (self.coords[0] - 8, self.coords[1] - 8), window)

    def check_click(self):
        return InputHandler.check_mouse_hover(self.rect)

    def __lt__(self, other):
        return int(self.id[1:]) < int(other.id[1:])


