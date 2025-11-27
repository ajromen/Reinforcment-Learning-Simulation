import pygame

from src.ui import colors
from src.ui.buttons_manager import ButtonsManager
from src.utils.constants import WINDOW_HEIGHT, FPS


class CreationScene:
    def __init__(self, window):
        self.window = window
        self.mode = "select" #select,muscle,joint,bone
        self.buttons = ButtonsManager(window)
        self.buttons.create_creation_scene_buttons()

    
    def start(self,preload=''):
        if preload!='':
            pass # load creature

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

            self.window.fill(colors.background_primary)
            rect_area = pygame.Rect(0, 0, 100, WINDOW_HEIGHT)

            # Draw a filled rectangle onto the window surface
            pygame.draw.rect(self.window, "#191919", rect_area)
            action = self.buttons.show_check_creation_scene_buttons(clicked)

            if action:
                pass


            clock.tick(FPS)
            pygame.display.flip()

                    
        return