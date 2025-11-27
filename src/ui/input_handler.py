import pygame

class InputHandler:
    @staticmethod
    def get_square(is_flipped=False, size=8):
        mouse_pos = pygame.mouse.get_pos()
        col = (mouse_pos[0] - 26) // 94
        row = (mouse_pos[1] - 26) // 94

        if col < 0 or col >= size or row < 0 or row >= size:
            return None, None

        if is_flipped:
            row = size - row - 1
            col = size - col - 1

        return row, col

    @staticmethod
    def check_mouse_hover(rect):
        mouse_pos = pygame.mouse.get_pos()
        return rect.collidepoint(mouse_pos)
