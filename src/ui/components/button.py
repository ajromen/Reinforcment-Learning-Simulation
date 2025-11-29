from pygame import Rect

from src.ui.input_handler import InputHandler


class Button:
    def __init__(self, x, y, width, height, image, image_inactive, name):
        self.pos = (x, y)
        self.rect = Rect(x, y, width, height)
        self.image = image
        self.image_inactive = image_inactive
        self.name = name
        self.active = False

    def check_show(self, window):
        if InputHandler.check_mouse_hover(self.rect):
            window.blit(self.image, (self.pos[0], self.pos[1]))
            return True

        if self.active:
            window.blit(self.image, (self.pos[0], self.pos[1]))
        else:
            window.blit(self.image_inactive, (self.pos[0], self.pos[1]))
        return False
