import pygame


class InputHandler:
    @staticmethod
    def coords_at_mouse():
        x, y = pygame.mouse.get_pos()
        mouse_pos = [x - 100, y]
        if mouse_pos[0] <= 0: return None

        col = (mouse_pos[0]) // 32
        row = (mouse_pos[1]) // 32

        return col, row

    @staticmethod
    def coords_to_pos(coords):
        print(coords)
        x, y = coords
        return x * 32 + 100 + 20 - 4, y * 32 + 20 - 4

    @staticmethod
    def check_mouse_hover(rect):
        mouse_pos = pygame.mouse.get_pos()
        return rect.collidepoint(mouse_pos)
