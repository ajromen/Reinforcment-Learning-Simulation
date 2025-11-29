from pygame import Vector2, gfxdraw
import pygame

from src.ui import colors
from src.ui.components.bone import Bone
from src.ui.components.joint import Joint
from src.ui.image_manager import ImageManager
from src.ui.input_handler import InputHandler
from src.utils.constants import UI_BONE_WIDTH, UI_MUSCLE_WIDTH


class Muscle:
    def __init__(self, id: str, bone1: Bone, bone2: Bone):
        self.id = id

        self.bone1 = min(bone1, bone2)
        self.bone2 = max(bone1, bone2)
        self.bone1.muscles.append(self)
        self.bone2.muscles.append(self)

        self.selected = False

    def show(self, window):
        a = Vector2(self.bone1.midpoint)
        b = Vector2(self.bone2.midpoint)

        d = b - a
        length = d.length()
        if length == 0:
            return

        angle = d.angle_to(Vector2(1, 0))
        img = ImageManager.muscle

        scaled = pygame.transform.smoothscale(img, (int(length), int(UI_MUSCLE_WIDTH)))

        rotated = pygame.transform.rotate(scaled, angle)

        mid = (a + b) * 0.5

        rect = rotated.get_rect(center=mid)

        if self.selected:
            n = Vector2(-d.y, d.x).normalize() * (UI_MUSCLE_WIDTH / 2)
            bw = 1.5
            p1 = a + n * bw
            p2 = a - n * bw
            p3 = b - n * bw
            p4 = b + n * bw
            gfxdraw.aapolygon(window, [p1, p2, p3, p4], colors.foreground_rgb)

        window.blit(rotated, rect)

    def check_click(self):
        return InputHandler.check_mouse_hover_rotated(Vector2(self.bone1.midpoint), Vector2(self.bone2.midpoint), UI_MUSCLE_WIDTH)


    @staticmethod
    def follow_mouse(a: tuple[float, float], window):
        a = Vector2(a)
        b = Vector2(pygame.mouse.get_pos())

        d = b - a

        n = Vector2(-d.y, d.x).normalize() * (UI_MUSCLE_WIDTH / 2)

        p1 = a + n
        p2 = a - n
        p3 = b - n
        p4 = b + n

        gfxdraw.filled_polygon(window, [p1, p2, p3, p4], colors.muscle_rgb_semi_transparent)
        gfxdraw.aapolygon(window, [p1, p2, p3, p4], colors.muscle_rgb_semi_transparent)

    @staticmethod
    def check_if_exists(b1: Bone, b2: Bone, muscles):
        bone1 = min(b1, b2)
        bone2 = max(b1, b2)

        for muscle in muscles:
            if bone1 == muscle.bone1 and  bone2 ==  muscle.bone2:
                return True

        return False
