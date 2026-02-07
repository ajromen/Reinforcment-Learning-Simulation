import pygame

from src.scenes.analysis_scene import AnalysisScene
from src.scenes.creation_scene import CreationScene
from src.ui.image_manager import ImageManager
from src.ui.text_renderer import TextRenderer
from src.ui.ui_settings import WINDOW_WIDTH, WINDOW_HEIGHT


def main():
    creature = None
    pygame.init()

    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    ImageManager.load_all()

    creation_scene = CreationScene(window, creature)
    creature = creation_scene.start()
    pygame.quit()
    TextRenderer.clear_all()
    ImageManager.clear_all()
    if creature is None:
        return
    creature, nn_layers = creature
    analysis_scene = AnalysisScene(creature,nn_layers)
    analysis_scene.start()


if __name__ == "__main__":
    main()
