from typing import List

import pygame
import pymunk
from pymunk import pygame_util

from src.agents.agent import Agent
from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.ui.colors import background_secondary, foreground, light_background, background_dots, background_primary, \
    foreground_secondary
from src.ui.text_renderer import TextRenderer
from src.utils.constants import FPS, WINDOW_WIDTH, WINDOW_HEIGHT, SIMULATION_SUBSTEPS, GROUND_Y, GRAVITY


class SimulationWindow:
    def __init__(self, creature: Creature, model: Agent):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.space = pymunk.Space()
        self.creature = CreaturePymunk(creature, self.space)
        self.draw = pygame_util.DrawOptions(self.window)
        self.setup_space()

        self.camera_offset = pygame.math.Vector2(0, 0)
        self.screen_center = pygame.math.Vector2(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

    def start(self):
        clock = pygame.time.Clock()
        random_muscle = self.creature.motors[0]
        running = True
        while running:
            clicked = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        random_muscle.rate = 4
                    elif event.key == pygame.K_DOWN:
                        random_muscle.rate = -4

                    if event.key == pygame.K_LEFT:
                        self.creature.move(-10)
                    elif event.key == pygame.K_RIGHT:
                        self.creature.move(10)

            self.run_pymunk()

            self.show()
            self.show_ui(clicked)

            # print(self.creature.get_center())

            self.model_step()

            clock.tick(FPS)
            pygame.display.flip()

    def setup_space(self):
        self.space.gravity = (0, GRAVITY)
        static_body = self.space.static_body
        self.ground = pymunk.Segment(static_body, (0, GROUND_Y), (WINDOW_WIDTH, GROUND_Y), 5.0)
        self.ground.friction = 1.0
        self.ground_1 = pymunk.Segment(static_body, (WINDOW_WIDTH, GROUND_Y), (WINDOW_WIDTH * 2, GROUND_Y), 5.0)
        self.space.add(self.ground_1)
        self.space.add(self.ground)

    def show_ui(self, clicked):
        pass

    def show(self):
        self.move_camera(self.creature.get_center())
        self.move_ground()

        self.window.fill(background_secondary)
        self.draw_ground_markers()
        self.space.debug_draw(self.draw)

    def draw_ground_markers(self):
        left = -self.camera_offset.x
        right = left + WINDOW_WIDTH
        y = GROUND_Y
        color = foreground_secondary

        for x in range(int(left // 50) * 50, int(right), 50):
            screen_x = x + self.camera_offset.x
            if x % 200 == 0:
                pygame.draw.line(self.window, color, (screen_x, y), (screen_x, y - 20), 2)
                TextRenderer.render_text(f"{x // 200 - 3} m", 16, color, (screen_x-10, y - 50), self.window)
            else:
                pygame.draw.line(self.window, color, (screen_x, y), (screen_x, y - 10), 1)

    def move_camera(self, center):
        target = pygame.math.Vector2(self.screen_center.x - center[0], 0)
        self.camera_offset = target
        tx, ty = target
        self.draw.transform = pymunk.Transform(a=1, b=0, c=0, d=1, tx=tx, ty=ty)


    def move_ground(self):
        left = -self.camera_offset.x
        # right = -self.camera_offset.x + WINDOW_WIDTH
        if self.ground.b.x < left:
            self.space.remove(self.ground)
            move = self.ground_1.b.x
            self.ground = pymunk.Segment(self.space.static_body, (move, GROUND_Y), (move + WINDOW_WIDTH, GROUND_Y),
                                         5.0)
            self.space.add(self.ground)

        elif self.ground_1.b.x < left:
            self.space.remove(self.ground_1)
            move = self.ground.b.x
            self.ground_1 = pymunk.Segment(self.space.static_body, (move, GROUND_Y), (move + WINDOW_WIDTH, GROUND_Y),
                                           5.0)
            self.space.add(self.ground_1)

    def run_pymunk(self):
        dt = 1 / FPS
        for _ in range(SIMULATION_SUBSTEPS):
            self.space.step(dt / 6)

    def model_step(self):
        pass
