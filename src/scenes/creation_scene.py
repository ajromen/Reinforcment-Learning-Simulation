import pygame

from src.ui import colors
from src.ui.buttons_manager import ButtonsManager
from src.ui.image_manager import ImageManager
from src.ui.input_handler import InputHandler
from src.ui.text_renderer import TextRenderer
from src.utils.constants import WINDOW_HEIGHT, FPS


class CreationScene:
    def __init__(self, window):
        self.window = window
        self.mode = "select"  # select,muscle,joint,bone
        self.buttons = ButtonsManager(window)
        self.buttons.create_creation_scene_buttons()

    def start(self, preload=''):
        if preload != '':
            pass  # load creature

        clock = pygame.time.Clock()
        pos = (0, 0)

        joints = []
        muscles = []
        bones = []

        running = True
        while running:
            clicked = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked = True

            self.window.fill(colors.background_primary)
            rect_area = pygame.Rect(0, 0, 100, WINDOW_HEIGHT)

            self.draw_grid()

            pygame.draw.circle(self.window, colors.background_secondary, pos, 20)
            TextRenderer.render_text("j1",13,"#ffffff",(pos[0]-8,pos[1]-8),self.window)


            pygame.draw.rect(self.window, colors.background_secondary, rect_area)
            action = self.buttons.show_check_creation_scene_buttons(clicked)

            if action:
                pass
            elif clicked:
                location = InputHandler.coords_at_mouse()
                if location: pos = InputHandler.coords_to_pos(location)


            clock.tick(FPS)
            pygame.display.flip()

        return

    def draw_grid(self):
        self.window.blit(ImageManager.dots_grid_small, (100, 0))
