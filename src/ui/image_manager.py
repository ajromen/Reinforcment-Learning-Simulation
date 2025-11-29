import pygame

class ImageManager:
    select_button = None
    joint_button = None
    bone_button = None
    muscle_button = None
    clear_button = None
    delete_button = None
    evolve_button = None
    neural_network_button = None
    load_button = None
    save_button = None

    select_button_inactive = None
    joint_button_inactive = None
    bone_button_inactive = None
    muscle_button_inactive = None
    clear_button_inactive = None
    delete_button_inactive = None
    evolve_button_inactive = None
    neural_network_button_inactive = None
    load_button_inactive = None
    save_button_inactive = None





    dots_grid_small = None
    muscle = None


    @staticmethod
    def load_all():
        ImageManager.select_button = pygame.image.load("assets/select_button.png").convert_alpha()
        ImageManager.joint_button = pygame.image.load("assets/joint_button.png").convert_alpha()
        ImageManager.bone_button = pygame.image.load("assets/bone_button.png").convert_alpha()
        ImageManager.clear_button = pygame.image.load("assets/clear_button.png").convert_alpha()
        ImageManager.muscle_button = pygame.image.load("assets/muscle_button.png").convert_alpha()
        ImageManager.delete_button = pygame.image.load("assets/delete_button.png").convert_alpha()
        ImageManager.evolve_button = pygame.image.load("assets/evolve_button.png").convert_alpha()
        ImageManager.neural_network_button = pygame.image.load("assets/neural_network_button.png").convert_alpha()
        ImageManager.load_button = pygame.image.load("assets/load_button.png").convert_alpha()
        ImageManager.save_button = pygame.image.load("assets/save_button.png").convert_alpha()

        ImageManager.select_button_inactive = pygame.image.load("assets/select_button_inactive.png").convert_alpha()
        ImageManager.joint_button_inactive = pygame.image.load("assets/joint_button_inactive.png").convert_alpha()
        ImageManager.bone_button_inactive = pygame.image.load("assets/bone_button_inactive.png").convert_alpha()
        ImageManager.clear_button_inactive = pygame.image.load("assets/clear_button_inactive.png").convert_alpha()
        ImageManager.muscle_button_inactive = pygame.image.load("assets/muscle_button_inactive.png").convert_alpha()
        ImageManager.delete_button_inactive = pygame.image.load("assets/delete_button_inactive.png").convert_alpha()
        ImageManager.evolve_button_inactive = pygame.image.load("assets/evolve_button_inactive.png").convert_alpha()
        ImageManager.neural_network_button_inactive = pygame.image.load("assets/neural_network_button_inactive.png").convert_alpha()
        ImageManager.load_button_inactive = pygame.image.load("assets/load_button_inactive.png").convert_alpha()
        ImageManager.save_button_inactive = pygame.image.load("assets/save_button_inactive.png").convert_alpha()


        ImageManager.dots_grid_small = pygame.image.load("assets/dots_grid.png").convert_alpha()
        ImageManager.muscle = pygame.image.load("assets/muscle.png").convert_alpha()




