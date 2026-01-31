import os

import pygame

from src.utils.constants import ASSETS_PATH


class TextRenderer:
    fonts = {}
    path = ASSETS_PATH+"JetBrainsMono-Regular.ttf"

    @staticmethod
    def load_font(size):
        if size in TextRenderer.fonts:
            return TextRenderer.fonts[size]

        TextRenderer.fonts[size] = pygame.font.Font(TextRenderer.path, size)
        return TextRenderer.fonts[size]

    @staticmethod
    def render_text(text: str, size: int, color: str, pos: tuple[float, float], window: pygame.Surface):
        font = TextRenderer.load_font(size)
        text_surface = font.render(text, True, color)
        window.blit(text_surface, pos)
