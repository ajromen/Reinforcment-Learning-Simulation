import pygame

class CreationScene:
    def __init__(self, window):
        self.window = window
        self.mode = "select" #select,muscle,joint,bone
    
    def start(preload=''):
        if preload!='':
            pass # load creature

        
        running = True
        while running:
            clicked = False
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked = True
            

                    
        return