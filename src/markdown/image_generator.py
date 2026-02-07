import pygame
from pygame import Vector2

from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.ui import colors
from src.ui.colors import bone_rgb, background_primary
from src.ui.image_manager import ImageManager
from src.ui.text_renderer import TextRenderer
from src.utils.creature_loader import CreatureLoader


class ImageGenerator:
    def __init__(self, creature: Creature, save_path: str):
        pass

    @staticmethod
    def generate_creature_image(creature: Creature, save_path: str):
        pygame.init()
        creature = creature
        joints, bones, muscles = CreatureLoader.creature_to_ui(creature)
        TextRenderer.clear_all()
        ImageManager.clear_all()
        window = pygame.display.set_mode((800, 600))
        ImageManager.load_for_simulation()
        save_path = save_path

        while True:
            window.fill(background_primary)
            window.blit(ImageManager.dots_grid_small, (0, 0))

            for muscle in muscles:
                muscle.show(window)

            for bone in bones:
                bone.show(window)

            for joint in joints:
                joint.show(window)

            pygame.display.flip()


