from typing import Dict

import pygame

from src.ui.components.bone import Bone
from src.ui import colors
from src.ui.buttons_manager import ButtonsManager
from src.ui.components.joint import Joint
from src.ui.image_manager import ImageManager
from src.ui.input_handler import InputHandler
from src.ui.text_renderer import TextRenderer
from src.utils.constants import WINDOW_HEIGHT, FPS


class CreationScene:
    def __init__(self, window):
        self.bones: Dict[str, Bone] = {}
        self.muscles: Dict[str, Joint] = {}
        self.joints: Dict[str, Joint] = {}
        self.window = window
        self.mode = "select"  # select,muscle,joint,bone
        self.mode_func = self.select_mode
        self.buttons = ButtonsManager(window)
        self.buttons.create_creation_scene_buttons()

        self.joint_num = 0
        self.muscle_num = 0
        self.bone_num = 0

        self.last_selected: Bone | Joint | None = None

        self.mode_names = {
            "select": self.select_mode,
            "joint": self.joint_mode,
            "muscle": self.muscle_mode,
            "bone": self.bone_mode,
            "delete": self.delete_mode,
            "evolve": self.evolve,
            "neural_network": self.neural_network_button,
            "clear": self.clear,
        }

    def start(self, preload=''):
        if preload != '':
            pass  # load creature

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

            action = self.show_ui(clicked)

            if action:
                self.mode = action
                self.mode_func = self.mode_names.get(action, lambda: print("Nije pronadjen mod"))
            # elif clicked:
            #     location = InputHandler.coords_at_mouse()
            #     if location: pos = InputHandler.coords_to_pos(location)

            self.mode_func(clicked)

            clock.tick(FPS)
            pygame.display.flip()

        return

    def show_ui(self, clicked):
        self.window.fill(colors.background_primary)
        self.draw_grid()
        rect_area = pygame.Rect(0, 0, 100, WINDOW_HEIGHT)
        pygame.draw.rect(self.window, colors.background_secondary, rect_area)
        action = self.buttons.show_check_creation_scene_buttons(clicked)

        # joints muscles and bones
        if self.mode == "bone" and self.last_selected is not None and self.last_selected.id[0] == 'j':
            Bone.follow_mouse(self.last_selected.coords, self.window)

        for muscle in self.muscles.values():
            pass

        for bone in self.bones.values():
            bone.show(self.window)

        for joint in self.joints.values():
            joint.show(self.window)

        # debug
        TextRenderer.render_text("Current Mode: " + self.mode, 13, "#ffffff", (100, 780), self.window)
        TextRenderer.render_text("Selected: " + str(self.last_selected), 13, "#ffffff", (100, 740), self.window)
        if self.last_selected:
            TextRenderer.render_text("Slected id: " + str(self.last_selected.id), 13, "#ffffff", (100, 720),
                                     self.window)
        TextRenderer.render_text("Current Mode: " + str(self.mode_func.__name__), 13, "#ffffff", (100, 760), self.window)
        return action

    def unselect(self):
        if self.last_selected: self.last_selected.selected = False
        self.last_selected = None

    def draw_grid(self):
        self.window.blit(ImageManager.dots_grid_small, (100, 0))

    def get_function_for_mode(self, clicked):
        pass

    def select_mode(self, clicked):
        if not clicked: return

        self.unselect()

        #joints
        for joint in self.joints.values():
            if joint.check_click():
                joint.selected = True
                self.last_selected = joint
                return

        #bones
        for bone in self.bones.values():
            if bone.check_click():
                bone.selected = True
                self.last_selected = bone
                return

    def joint_mode(self, clicked):
        if not clicked: return
        location = InputHandler.coords_at_mouse()
        if not location: return

        for joint in self.joints.values():
            if joint.pos == location:
                return

        pos = InputHandler.coords_to_pos(location)

        id = "j" + str(self.joint_num)
        self.joint_num += 1
        self.joints[id] = Joint(pos, location, id)

    def bone_mode(self, clicked):
        if not clicked: return

        if self.last_selected and self.last_selected.id[0] != "j":
            self.unselect()

        if self.last_selected is None:
            for joint in self.joints.values():
                if joint.check_click():
                    joint.selected = True
                    self.last_selected = joint
                    return
            return

        for joint in self.joints.values():
            if joint.check_click():
                if Bone.check_if_exists(self.last_selected, joint,self.bones.values()): return
                if self.last_selected == joint:
                    self.unselect()
                    return
                print("bone created")
                id = "b" + str(self.bone_num)
                self.bone_num += 1
                b = Bone(id, self.last_selected, joint)
                self.unselect()
                self.bones[id] = b
                return

    def muscle_mode(self, clicked):
        pass

    def delete_mode(self, clicked):
        if not clicked: return
        if self.last_selected is None:
            self.mode = 'select'
            self.mode_func = self.select_mode
            return

        id: str = self.last_selected.id
        self.last_selected = None
        part = id[0]
        if part == 'j':
            del (self.joints[id])
        if part == 'm':
            del (self.muscles[id])
        if part == 'b':
            del (self.bones[id])

        self.mode = 'select'
        self.mode_func = self.select_mode

    def evolve(self, clicked):
        pass

    def neural_network_button(self, clicked):
        pass

    def clear(self, clicked):
        pass
