import os

import pygame

class TextRenderer:
    fonts = {}
    path = "assets/JetBrainsMono-Regular.ttf"

    @staticmethod
    def load_font(size):
        if size in TextRenderer.fonts:
            return TextRenderer.fonts[size]

        TextRenderer.fonts[size] = pygame.font.Font(TextRenderer.path, size)
        return TextRenderer.fonts[size]

    @staticmethod
    def render_text(text, size, color, pos, window):
        font = TextRenderer.load_font(size)
        text_surface = font.render(text, True, color)
        window.blit(text_surface, pos)