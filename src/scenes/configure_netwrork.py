import pygame
from pygame import Rect, Vector2

from src.ui import colors
from src.ui.input_handler import InputHandler
from src.ui.text_renderer import TextRenderer
from src.utils.constants import FPS, RADIUS_NEURONS, W_SPACING_NEURONS, H_SPACING_NEURONS


class ConfigureNetwork:
    def __init__(self, window, depth, layer_widths):
        self.window = window
        self.depth = depth
        self.layer_widths = layer_widths

    def start(self):
        clock = pygame.time.Clock()

        background_rect = Rect(150, 100, 900, 600)
        exit_rect = Rect(1007, 105, 37, 37)

        while True:
            clicked = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked = True

            pygame.draw.rect(self.window, colors.background_secondary, background_rect, border_radius=5)
            pygame.draw.rect(self.window, colors.foreground_secondary, background_rect, width=1, border_radius=5)
            pygame.draw.line(self.window, colors.foreground, (1020, 118), (1031, 130), 2)
            pygame.draw.line(self.window, colors.foreground, (1031, 118), (1020, 130), 2)
            TextRenderer.render_text("Configure neural network", 16, colors.foreground, (168, 118), self.window)

            # exit condition
            if clicked and InputHandler.check_mouse_hover(exit_rect):
                return self.depth, self.layer_widths

            w, h, rect = self.recalculate_dimensions()
            self.show_network(w, h, rect)

            self.check_mouse(clicked,w,h,rect)

            clock.tick(FPS)
            pygame.display.flip()

    def recalculate_dimensions(self):
        w = self.depth * (RADIUS_NEURONS * 2 + W_SPACING_NEURONS)
        h = (max(self.layer_widths)) * (RADIUS_NEURONS * 2 + H_SPACING_NEURONS)

        rect = Rect(150 + 900 / 2 - w / 2, 100 + 600 / 2 - h / 2, w, h)
        return w, h, rect

    def check_mouse(self,clicked,w,h,rect):
        if not InputHandler.check_mouse_hover(rect): return
        # left = rect.left + (RADIUS_NEURONS * 2 + W_SPACING_NEURONS) / 2
        mouse_pos = pygame.mouse.get_pos()
        col_width = RADIUS_NEURONS * 2 + W_SPACING_NEURONS

        for i in range(self.depth - 1):  # between layer i and i+1
            gap_x = rect.left + col_width * (i + 1) - W_SPACING_NEURONS / 2
            gap_rect = Rect(gap_x, rect.top, W_SPACING_NEURONS, rect.height)

            # debug: visualize gap zone
            if gap_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.window, (113, 176, 129), gap_rect)  # green outline

    def show_network(self, w, h, rect):
        # pygame.draw.rect(self.window, colors.foreground, rect, width=1, border_radius=5)
        ball_positions: list[list[tuple]] = []
        for i in range(self.depth):
            h_top = (self.layer_widths[i] - 1) * (H_SPACING_NEURONS * 2 + RADIUS_NEURONS)
            start_y = (h - h_top) / 2
            layer_positions = []
            for j in range(self.layer_widths[i]):
                hor = rect.left + (RADIUS_NEURONS * 2 + W_SPACING_NEURONS) * i + (
                        RADIUS_NEURONS * 2 + W_SPACING_NEURONS) / 2
                vert = rect.top + start_y + (H_SPACING_NEURONS * 2 + RADIUS_NEURONS) * j
                layer_positions.append((hor, vert))
            ball_positions.append(layer_positions)

        for i in range(len(ball_positions) - 1):
            for j in range(len(ball_positions[i])):
                for k in range(len(ball_positions[i + 1])):
                    pygame.draw.aaline(self.window, colors.foreground, ball_positions[i][j], ball_positions[i + 1][k],
                                     width=2)

        for layer in ball_positions:
            for (hor, vert) in layer:
                pygame.draw.circle(self.window, colors.foreground_secondary, center=(hor, vert), radius=RADIUS_NEURONS)
