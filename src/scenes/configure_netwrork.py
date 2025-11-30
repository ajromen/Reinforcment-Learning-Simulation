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
        self.plus_buttons = []
        self.minus_buttons = []

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

            self.check_mouse(clicked, rect)

            clock.tick(FPS)
            pygame.display.flip()

    def recalculate_dimensions(self):
        w = self.depth * (RADIUS_NEURONS * 2 + W_SPACING_NEURONS)
        h = (max(self.layer_widths) - 1) * (RADIUS_NEURONS * 2 + H_SPACING_NEURONS)

        rect = Rect(150 + 900 / 2 - w / 2, 100 + 600 / 2 - h / 2, w, h)
        return w, h, rect

    def check_mouse(self, clicked, rect):

        mouse_pos = pygame.mouse.get_pos()
        col_width = RADIUS_NEURONS * 2 + W_SPACING_NEURONS
        plus_color = colors.foreground
        minus_color = colors.foreground

        for i in range(1, len(self.minus_buttons) - 1):
            curr_rect = self.minus_buttons[i]
            pygame.draw.rect(self.window, minus_color, curr_rect, width=1, border_radius=5)
            # minus sign
            pygame.draw.line(self.window, minus_color, (curr_rect.left + 5, curr_rect.top / 2 + curr_rect.bottom / 2),
                             (curr_rect.right - 5, curr_rect.top / 2 + curr_rect.bottom / 2), width=1)
            if not clicked: continue
            if not curr_rect.collidepoint(mouse_pos): continue
            if self.layer_widths[i] <= 1: continue
            self.layer_widths[i] -= 1

        for i in range(1, len(self.plus_buttons) - 1):
            curr_rect = self.plus_buttons[i]
            pygame.draw.rect(self.window, plus_color, curr_rect, width=1, border_radius=5)
            # plus sign
            pygame.draw.line(self.window, minus_color, (curr_rect.left / 2 + curr_rect.right / 2, curr_rect.top + 5),
                             (curr_rect.left / 2 + curr_rect.right / 2, curr_rect.bottom - 5), width=1)
            pygame.draw.line(self.window, minus_color, (curr_rect.left + 5, curr_rect.top / 2 + curr_rect.bottom / 2),
                             (curr_rect.right - 5, curr_rect.top / 2 + curr_rect.bottom / 2), width=1)

            if not clicked: continue
            if not curr_rect.collidepoint(mouse_pos): continue
            if self.layer_widths[i] >= 11: continue
            self.layer_widths[i] += 1

        if not InputHandler.check_mouse_hover(rect): return
        for i in range(self.depth - 1):
            gap_x = rect.left + col_width * (i + 1) - W_SPACING_NEURONS / 2
            layer_x = rect.left + (RADIUS_NEURONS * 2 + W_SPACING_NEURONS) / 2 - RADIUS_NEURONS + col_width * (i + 1)
            gap_rect = Rect(gap_x, rect.top, W_SPACING_NEURONS, rect.height)
            layer_rect = Rect(layer_x, rect.top, RADIUS_NEURONS * 2, rect.height)

            if layer_rect.collidepoint(mouse_pos) and i != self.depth - 2:
                pygame.draw.rect(self.window, (176, 113, 129), layer_rect, width=6, border_radius=5)
                if not clicked: continue
                if self.depth <= 3: continue
                mouse_x = mouse_pos[0]
                idx = (mouse_x - rect.left - (RADIUS_NEURONS * 2 + W_SPACING_NEURONS) / 2 - RADIUS_NEURONS) // col_width
                del self.layer_widths[int(idx + 1)]
                self.depth -= 1

            elif gap_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.window, (113, 176, 129), gap_rect, width=6, border_radius=5)
                if not clicked: continue
                if self.depth >= 9: continue
                mouse_x = mouse_pos[0]
                idx = (mouse_x - rect.left - (RADIUS_NEURONS * 2 + W_SPACING_NEURONS) / 2 - RADIUS_NEURONS) // col_width
                self.layer_widths.insert(int(idx + 1), 10)
                self.depth += 1

    def show_network(self, w, h, rect):
        # pygame.draw.rect(self.window, colors.foreground, rect, width=1, border_radius=5)

        # calculating ball plus minus buttons positions
        ball_positions: list[list[tuple]] = []
        self.plus_buttons = []
        self.minus_buttons = []
        for i in range(self.depth):
            h_top = (self.layer_widths[i] - 1) * (H_SPACING_NEURONS * 2 + RADIUS_NEURONS)
            start_y = (h - h_top) / 2
            layer_positions = []
            hor = rect.left + (RADIUS_NEURONS * 2 + W_SPACING_NEURONS) * i + (
                    RADIUS_NEURONS * 2 + W_SPACING_NEURONS) / 2
            plus_button_rect = Rect(hor - RADIUS_NEURONS, rect.top - 10 - RADIUS_NEURONS * 2, RADIUS_NEURONS * 2,
                                    RADIUS_NEURONS * 2)
            minus_button_rect = Rect(hor - RADIUS_NEURONS, rect.bottom + 10, RADIUS_NEURONS * 2,
                                     RADIUS_NEURONS * 2)
            self.plus_buttons.append(plus_button_rect)
            self.minus_buttons.append(minus_button_rect)

            for j in range(self.layer_widths[i]):
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
