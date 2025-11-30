import pygame
from pygame import Vector2


class InputHandler:
    @staticmethod
    def coords_at_mouse() -> tuple[int, int] | None:
        x, y = pygame.mouse.get_pos()
        mouse_pos = [x - 100, y]
        if mouse_pos[0] <= 0: return None

        col = (mouse_pos[0]) // 32
        row = (mouse_pos[1]) // 32

        return col, row

    @staticmethod
    def coords_to_pos(coords: tuple[int, int]):
        x, y = coords
        return x * 32 + 100 + 20 - 4, y * 32 + 20 - 4

    @staticmethod
    def check_mouse_hover(rect: pygame.Rect):
        mouse_pos = pygame.mouse.get_pos()
        return rect.collidepoint(mouse_pos)

    @staticmethod
    def check_mouse_hover_rotated(a: Vector2, b:Vector2, width: int):
        mouse = pygame.Vector2(pygame.mouse.get_pos())

        ab = b - a
        am = mouse - a

        t = am.dot(ab) / ab.length_squared()

        t = max(0, min(1, t))

        closest = a + ab * t
        dist = (mouse - closest).length()

        return dist <= width/2