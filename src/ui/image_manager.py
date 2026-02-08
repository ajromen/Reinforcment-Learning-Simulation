import pygame

from src.ui.colors import IS_LIGHT_MODE
from src.utils.constants import ASSETS_PATH


class ImageManager:
    select_button = None
    joint_button = None
    bone_button = None
    muscle_button = None
    clear_button = None
    delete_button = None
    learn_button = None
    neural_network_button = None
    load_button = None
    save_button = None
    continue_button = None

    select_button_inactive = None
    joint_button_inactive = None
    bone_button_inactive = None
    muscle_button_inactive = None
    clear_button_inactive = None
    delete_button_inactive = None
    learn_button_inactive = None
    neural_network_button_inactive = None
    load_button_inactive = None
    save_button_inactive = None
    continue_button_inactive = None

    dots_grid_small = None
    muscle = None


    @staticmethod
    def load_all():
        ImageManager.select_button = pygame.image.load(ASSETS_PATH+"select_button.png").convert_alpha()
        ImageManager.joint_button = pygame.image.load(ASSETS_PATH+"joint_button.png").convert_alpha()
        ImageManager.bone_button = pygame.image.load(ASSETS_PATH+"bone_button.png").convert_alpha()
        ImageManager.clear_button = pygame.image.load(ASSETS_PATH+"clear_button.png").convert_alpha()
        ImageManager.muscle_button = pygame.image.load(ASSETS_PATH+"muscle_button.png").convert_alpha()
        ImageManager.delete_button = pygame.image.load(ASSETS_PATH+"delete_button.png").convert_alpha()
        ImageManager.learn_button = pygame.image.load(ASSETS_PATH+"learn_button.png").convert_alpha()
        ImageManager.neural_network_button = pygame.image.load(ASSETS_PATH+"neural_network_button.png").convert_alpha()
        ImageManager.load_button = pygame.image.load(ASSETS_PATH+"load_button.png").convert_alpha()
        ImageManager.save_button = pygame.image.load(ASSETS_PATH+"save_button.png").convert_alpha()
        ImageManager.continue_button = pygame.image.load(ASSETS_PATH+"continue_button.png").convert_alpha()

        ImageManager.select_button_inactive = pygame.image.load(ASSETS_PATH+"select_button_inactive.png").convert_alpha()
        ImageManager.joint_button_inactive = pygame.image.load(ASSETS_PATH+"joint_button_inactive.png").convert_alpha()
        ImageManager.bone_button_inactive = pygame.image.load(ASSETS_PATH+"bone_button_inactive.png").convert_alpha()
        ImageManager.clear_button_inactive = pygame.image.load(ASSETS_PATH+"clear_button_inactive.png").convert_alpha()
        ImageManager.muscle_button_inactive = pygame.image.load(ASSETS_PATH+"muscle_button_inactive.png").convert_alpha()
        ImageManager.delete_button_inactive = pygame.image.load(ASSETS_PATH+"delete_button_inactive.png").convert_alpha()
        ImageManager.learn_button_inactive = pygame.image.load(ASSETS_PATH+"learn_button_inactive.png").convert_alpha()
        ImageManager.neural_network_button_inactive = pygame.image.load(ASSETS_PATH+"neural_network_button_inactive.png").convert_alpha()
        ImageManager.load_button_inactive = pygame.image.load(ASSETS_PATH+"load_button_inactive.png").convert_alpha()
        ImageManager.save_button_inactive = pygame.image.load(ASSETS_PATH+"save_button_inactive.png").convert_alpha()
        ImageManager.continue_button_inactive = pygame.image.load(ASSETS_PATH+"continue_button_inactive.png").convert_alpha()

        ImageManager.dots_grid_small = pygame.image.load(ASSETS_PATH+"dots_grid.png").convert_alpha()

        if IS_LIGHT_MODE:
            ImageManager.invert_all()
        ImageManager.muscle = pygame.image.load(ASSETS_PATH+"muscle.png").convert_alpha()

    @staticmethod
    def invert_all():
        ImageManager.select_button = invert_image(ImageManager.select_button)
        ImageManager.joint_button = invert_image(ImageManager.joint_button)
        ImageManager.bone_button = invert_image(ImageManager.bone_button)
        ImageManager.muscle_button = invert_image(ImageManager.muscle_button)
        ImageManager.clear_button = invert_image(ImageManager.clear_button)
        ImageManager.delete_button = invert_image(ImageManager.delete_button)
        ImageManager.learn_button = invert_image(ImageManager.learn_button)
        ImageManager.neural_network_button = invert_image(ImageManager.neural_network_button)
        ImageManager.load_button = invert_image(ImageManager.load_button)
        ImageManager.save_button = invert_image(ImageManager.save_button)
        ImageManager.continue_button = invert_image(ImageManager.continue_button)

        ImageManager.select_button_inactive = invert_image(ImageManager.select_button_inactive)
        ImageManager.joint_button_inactive = invert_image(ImageManager.joint_button_inactive)
        ImageManager.bone_button_inactive = invert_image(ImageManager.bone_button_inactive)
        ImageManager.muscle_button_inactive = invert_image(ImageManager.muscle_button_inactive)
        ImageManager.clear_button_inactive = invert_image(ImageManager.clear_button_inactive)
        ImageManager.delete_button_inactive = invert_image(ImageManager.delete_button_inactive)
        ImageManager.learn_button_inactive = invert_image(ImageManager.learn_button_inactive)
        ImageManager.neural_network_button_inactive = invert_image(ImageManager.neural_network_button_inactive)
        ImageManager.load_button_inactive = invert_image(ImageManager.load_button_inactive)
        ImageManager.save_button_inactive = invert_image(ImageManager.save_button_inactive)
        ImageManager.continue_button_inactive = invert_image(ImageManager.continue_button_inactive)

        ImageManager.dots_grid_small = invert_image(ImageManager.dots_grid_small)

    @staticmethod
    def clear_all():
        ImageManager.select_button = None
        ImageManager.joint_button = None
        ImageManager.bone_button = None
        ImageManager.muscle_button = None
        ImageManager.clear_button = None
        ImageManager.delete_button = None
        ImageManager.learn_button = None
        ImageManager.neural_network_button = None
        ImageManager.load_button = None
        ImageManager.save_button = None

        ImageManager.select_button_inactive = None
        ImageManager.joint_button_inactive = None
        ImageManager.bone_button_inactive = None
        ImageManager.muscle_button_inactive = None
        ImageManager.clear_button_inactive = None
        ImageManager.delete_button_inactive = None
        ImageManager.learn_button_inactive = None
        ImageManager.neural_network_button_inactive = None
        ImageManager.load_button_inactive = None
        ImageManager.save_button_inactive = None

        ImageManager.dots_grid_small = None
        ImageManager.muscle = None

    @staticmethod
    def load_for_simulation():
        ImageManager.dots_grid_small = pygame.image.load(ASSETS_PATH + "dots_grid.png").convert_alpha()

        if IS_LIGHT_MODE:
            ImageManager.dots_grid_small = invert_image(ImageManager.dots_grid_small)

        ImageManager.muscle = pygame.image.load(ASSETS_PATH + "muscle.png").convert_alpha()


def invert_image(surface: pygame.Surface) -> pygame.Surface:
    inverted = surface.copy()

    arr = pygame.surfarray.pixels3d(inverted)
    arr[:] = 255 - arr

    del arr
    return inverted
