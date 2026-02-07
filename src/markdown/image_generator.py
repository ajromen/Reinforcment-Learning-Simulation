import os

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
    @staticmethod
    def generate_creature_image(creature: Creature, save_path: str):
        if os.path.exists(save_path):
            return
        pygame.init()
        TextRenderer.clear_all()
        ImageManager.clear_all()

        joints, bones, muscles = CreatureLoader.creature_to_ui(creature)

        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        for joint in joints:
            x, y = joint.coords
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        p = 30
        width = max_x - min_x + 2 * p
        height = max_y - min_y + 2 * p

        target_ratio = 4 / 3
        if width / height > target_ratio:
            height = int(width / target_ratio)
        else:
            width = int(height * target_ratio)

        window = pygame.display.set_mode((width, height), pygame.SRCALPHA)
        ImageManager.load_for_simulation()

        offset_x = p - min_x + (width - (max_x - min_x + 2 * p)) // 2
        offset_y = p - min_y + (height - (max_y - min_y + 2 * p)) // 2

        surface = pygame.Surface((width, height))
        surface.fill(background_primary)
        if ImageManager.dots_grid_small:
            surface.blit(ImageManager.dots_grid_small, (0, 0))

        for joint in joints:
            joint.coords = Vector2(joint.coords) + Vector2(offset_x, offset_y)
            joint.show(surface)

        for bone in bones:
            a = Vector2(bone.joint1.coords)
            b = Vector2(bone.joint2.coords)
            bone.midpoint = ((a.x + b.x) / 2, (a.y + b.y) / 2)

        if ImageManager.dots_grid_small:
            surface.blit(ImageManager.dots_grid_small, (0, 0))

        for muscle in muscles:
            muscle.show(surface)

        for bone in bones:
            bone.show(surface)

        for joint in joints:
            joint.show(surface)

        pygame.image.save(surface, save_path)
        pygame.quit()
        ImageManager.clear_all()
        TextRenderer.clear_all()
