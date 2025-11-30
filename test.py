import sys
import math
import numpy as np
import pygame
# import pymunk
# from pymunk import pygame_util
from src.scenes.creation_scene import CreationScene
from src.ui.image_manager import ImageManager
from src.utils.constants import WINDOW_WIDTH, WINDOW_HEIGHT

pygame.init()

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

ImageManager.load_all()

creation_scene = CreationScene(window)
creation_scene.start()

