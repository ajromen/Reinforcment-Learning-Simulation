import math

from pygame import Vector2, gfxdraw
import pygame
from pygame.typing import SequenceLike

from src.ui import colors
from src.ui.components.joint import Joint
from src.ui.input_handler import InputHandler
from src.utils.constants import UI_BONE_WIDTH


class Bone:
    def __init__(self, id: str, joint1: Joint, joint2: Joint):
        self.id = id

        self.joint1 = min(joint1, joint2)
        self.joint2 = max(joint1, joint2)
        self.joint1.bones.append(self)
        self.joint2.bones.append(self)

        self.selected = False

    def show(self, window):
        a = Vector2(self.joint1.coords)
        b = Vector2(self.joint2.coords)

        d = b - a

        n = Vector2(-d.y, d.x).normalize() * (UI_BONE_WIDTH / 2)

        if self.selected:
            bw = 1.5
            p1 = a + n * bw
            p2 = a - n * bw
            p3 = b - n * bw
            p4 = b + n * bw
            gfxdraw.aapolygon(window, [p1, p2, p3, p4], colors.foreground_rgb)

        p1 = a + n
        p2 = a - n
        p3 = b - n
        p4 = b + n

        gfxdraw.filled_polygon(window, [p1, p2, p3, p4], colors.bone_rgb)
        gfxdraw.aapolygon(window, [p1, p2, p3, p4], colors.bone_rgb)

    def check_click(self):
        return InputHandler.check_mouse_hover_rotated(Vector2(self.joint1.coords), Vector2(self.joint2.coords), UI_BONE_WIDTH)


    @staticmethod
    def follow_mouse(a: tuple[float, float], window):
        a = Vector2(a)
        b = Vector2(pygame.mouse.get_pos())

        d = b - a

        n = Vector2(-d.y, d.x).normalize() * (UI_BONE_WIDTH / 2)

        p1 = a + n
        p2 = a - n
        p3 = b - n
        p4 = b + n

        gfxdraw.filled_polygon(window, [p1, p2, p3, p4], colors.bone_rbg_semi_transparent)
        gfxdraw.aapolygon(window, [p1, p2, p3, p4], colors.bone_rbg_semi_transparent)

    @staticmethod
    def check_if_exists(j1: Joint, j2: Joint, bones):
        joint1 = min(j1, j2)
        joint2 = max(j1, j2)

        for bone in bones:
            if joint1 == bone.joint1 and  joint2 ==  bone.joint2:
                return True

        return False
