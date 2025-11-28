from pygame import Rect

from src.ui.input_handler import InputHandler


class Button:
    def __init__(self, x, y, width, height, image, name):
        self.pos = (x, y)
        self.rect = Rect(x, y, width, height)
        self.image = image
        self.name = name

    def check_show(self, window):
        if InputHandler.check_mouse_hover(self.rect):
            # window.blit
            #add highlight when hovered or diferent image
            window.blit(self.image, (self.pos[0], self.pos[1]))
            return True
        window.blit(self.image, (self.pos[0], self.pos[1]))
        return False
