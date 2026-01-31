from typing import List

import pygame
import pymunk
from pymunk import pygame_util

from src.agents.agent import Agent
from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.ui.colors import background_secondary
from src.utils.constants import FPS, WINDOW_WIDTH, WINDOW_HEIGHT, SIMULATION_SUBSTEPS


class SimulationWindow:
    def __init__(self, creature: Creature, model: Agent):
        pygame.init()
        self.window =  pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.space  = pymunk.Space()
        self.creature = CreaturePymunk(creature,self.space)
        self.draw = pygame_util.DrawOptions(self.window)
        self.setup_space()


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

            self.run_pymunk()
            self.show_ui(clicked)

            self.model_step()

            clock.tick(FPS)
            pygame.display.flip()

    def setup_space(self):
        self.space.gravity = (0, 1200)
        static_body = self.space.static_body
        ground = pymunk.Segment(static_body, (0, 600), (WINDOW_WIDTH, 600), 5.0)
        ground.friction = 1.0
        self.space.add(ground)

    def show_ui(self, clicked):
        pass

    def run_pymunk(self):
        dt = 1 / FPS
        for _ in range(SIMULATION_SUBSTEPS):
            self.space.step(dt / 6)

        self.window.fill(background_secondary)
        self.space.debug_draw(self.draw)

    def model_step(self):
        pass