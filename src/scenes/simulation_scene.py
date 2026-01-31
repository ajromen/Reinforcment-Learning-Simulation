import pygame
import pymunk

from src.models.creature import Creature
from src.utils.constants import FPS, SIMULATION_SUBSTEPS
from src.pymunk.creature_pymunk import CreaturePymunk


class SimulationScene:
    def __init__(self, creature: Creature):
        self.space = pymunk.Space()
        self.pymunk_creature = CreaturePymunk(creature)
        self.start()

    def start(self):
        clock = pygame.time.Clock()

        running = True
        while running:
            clicked = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked = True

            dt = 1.0 / FPS
            for _ in range(SIMULATION_SUBSTEPS):
                self.space.step(dt / SIMULATION_SUBSTEPS)

            clock.tick(FPS)
            pygame.display.flip()