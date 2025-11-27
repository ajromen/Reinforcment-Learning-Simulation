import pygame

class ImageManager:
    select_button = None
    joint_button = None
    bone_button = None
    muscle_button = None
    clear_button = None
    delete_button = None
    evolve_button = None
    dots_background = None
    neural_network_button = None

    @staticmethod
    def load_all():
        ImageManager.tabla = pygame.image.load("assets/select_button.png").convert_alpha()
