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
        ImageManager.select_button = pygame.image.load("assets/select_button.png").convert_alpha()
        #ImageManager.joint_button = pygame.image.load("assets/joint_button.png").convert_alpha()
        ImageManager.bone_button = pygame.image.load("assets/bone_button.png").convert_alpha()
        ImageManager.clear_button = pygame.image.load("assets/clear_button.png").convert_alpha()
        ImageManager.muscle_button = pygame.image.load("assets/muscle_button.png").convert_alpha()
        ImageManager.delete_button = pygame.image.load("assets/delete_button.png").convert_alpha()
        ImageManager.evolve_button = pygame.image.load("assets/evolve_button.png").convert_alpha()
        ImageManager.neural_network_button = pygame.image.load("assets/neural_network_button.png").convert_alpha()
        #ImageManager.dots_background = pygame.image.load("assets/dots_background.png").convert_alpha()




