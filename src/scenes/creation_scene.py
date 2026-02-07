import os
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Dict, List

import pygame
from sympy.logic.boolalg import Boolean

from src.models.creature import Creature
from src.scenes.configure_netwrork import ConfigureNetwork
from src.ui import colors
from src.ui.buttons_manager import ButtonsManager
from src.ui.components.bone import Bone
from src.ui.components.joint import Joint
from src.ui.components.muscle import Muscle
from src.ui.image_manager import ImageManager
from src.ui.input_handler import InputHandler
from src.ui.text_renderer import TextRenderer
from src.ui.ui_settings import WINDOW_WIDTH, FPS, WINDOW_HEIGHT
from src.utils.creature_loader import CreatureLoader


class CreationScene:
    def __init__(self, window, creature):
        self.bones: Dict[str, Bone] = {}
        self.muscles: Dict[str, Muscle] = {}
        self.joints: Dict[str, Joint] = {}
        self.window = window
        self.mode = "select"  # select,muscle,joint,bone
        self.mode_func = self.select_mode
        self.buttons = ButtonsManager(window)
        self.buttons.create_creation_scene_buttons()

        self.joint_num = 0
        self.muscle_num = 0
        self.bone_num = 0

        # network conf
        self.depth = 6
        self.layer_widths = [1, 7, 10, 10, 3, 1]

        self.last_selected: Bone | Joint | None = None

        self.mode_names = {
            "select": self.select_mode,
            "joint": self.joint_mode,
            "muscle": self.muscle_mode,
            "bone": self.bone_mode,
            "delete": self.delete_mode,
            "learn": self.learn,
            "neural_network": self.neural_network_button,
            "clear": self.clear,
            "load": self.load,
            "save": self.save,
            "continue": self.continue_func
        }

        if creature is not None:
            self.creature_to_ui(creature)

    # returns creature model with neural net configuration
    def start(self) -> tuple[Creature, List[int], bool] | None:
        self.switch_to_select()
        clock = pygame.time.Clock()

        running = True
        while running:
            clicked = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_j:
                        self.mode = 'joint'
                        self.mode_func = self.mode_names.get(self.mode, lambda: print("Nije pronadjen mod"))
                    elif event.key == pygame.K_m:
                        self.mode = 'muscle'
                        self.mode_func = self.mode_names.get(self.mode, lambda: print("Nije pronadjen mod"))
                    elif event.key == pygame.K_b:
                        self.mode = 'bone'
                        self.mode_func = self.mode_names.get(self.mode, lambda: print("Nije pronadjen mod"))
                    elif event.key == pygame.K_c:
                        self.mode = 'clear'
                        self.mode_func = self.mode_names.get(self.mode, lambda: print("Nije pronadjen mod"))

            action = self.show_ui(clicked)

            if action:
                self.mode = action
                self.mode_func = self.mode_names.get(action, lambda: print("Nije pronadjen mod"))

            if self.mode == 'learn':
                creature = CreatureLoader.ui_to_creature(self.joints.values(), self.bones.values(),
                                                         self.muscles.values())
                if creature is None:
                    self.print_notif(False)
                else:
                    CreatureLoader.save_creature(creature)
                    return creature, self.layer_widths, False
                self.switch_to_select()
                clicked = False

            if self.mode == 'continue':
                creature = CreatureLoader.ui_to_creature(self.joints.values(), self.bones.values(),
                                                         self.muscles.values())
                if creature is None:
                    return None
                return creature, self.layer_widths, True

            self.mode_func(clicked)

            clock.tick(FPS)
            pygame.display.flip()

        return None

    def show_ui(self, clicked):
        self.window.fill(colors.background_primary)
        self.draw_grid()
        rect_area = pygame.Rect(0, 0, 100, WINDOW_HEIGHT)
        pygame.draw.rect(self.window, colors.background_secondary, rect_area)
        action = self.buttons.show_check_creation_scene_buttons(clicked, self.mode)

        # joints muscles and bones
        if self.mode == "bone" and self.last_selected is not None and self.last_selected.id[0] == 'j':
            Bone.follow_mouse(self.last_selected.coords, self.window)

        if self.mode == "muscle" and self.last_selected is not None and self.last_selected.id[0] == 'b':
            Muscle.follow_mouse(self.last_selected.midpoint, self.window)

        for muscle in self.muscles.values():
            muscle.show(self.window)

        for bone in self.bones.values():
            bone.show(self.window)

        for joint in self.joints.values():
            joint.show(self.window)

        # debug
        TextRenderer.render_text("Current Mode: " + self.mode, 13, "#ffffff", (100, 780), self.window)
        TextRenderer.render_text("Selected: " + str(self.last_selected), 13, "#ffffff", (100, 740), self.window)
        if self.last_selected:
            TextRenderer.render_text("Selected id: " + str(self.last_selected.id), 13, "#ffffff", (100, 720),
                                     self.window)
        TextRenderer.render_text("Current Mode: " + str(self.mode_func.__name__), 13, "#ffffff", (100, 760),
                                 self.window)
        return action

    def unselect(self):
        if self.last_selected: self.last_selected.selected = False
        self.last_selected = None

    def draw_grid(self):
        self.window.blit(ImageManager.dots_grid_small, (100, 0))

    def select_mode(self, clicked):
        if not clicked: return

        self.unselect()

        # joints
        for joint in self.joints.values():
            if joint.check_click():
                joint.selected = True
                self.last_selected = joint
                return

        # bones
        for bone in self.bones.values():
            if bone.check_click():
                bone.selected = True
                self.last_selected = bone
                return

        # bones
        for muscle in self.muscles.values():
            if muscle.check_click():
                muscle.selected = True
                self.last_selected = muscle
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
                if Bone.check_if_exists(self.last_selected, joint, self.bones.values()): return
                if self.last_selected == joint:
                    self.unselect()
                    return
                id = "b" + str(self.bone_num)
                self.bone_num += 1
                b = Bone(id, self.last_selected, joint)
                self.unselect()
                self.bones[id] = b
                return

    def muscle_mode(self, clicked):
        if not clicked: return

        if self.last_selected and self.last_selected.id[0] != "b":
            self.unselect()

        if self.last_selected is None:
            for bone in self.bones.values():
                if bone.check_click():
                    bone.selected = True
                    self.last_selected = bone
                    return
            return

        for bone in self.bones.values():
            if bone.check_click():
                if Muscle.check_if_exists(self.last_selected, bone, self.muscles.values()): return
                if self.last_selected == bone:
                    self.unselect()
                    return
                id = "m" + str(self.muscle_num)
                self.muscle_num += 1
                m = Muscle(id, self.last_selected, bone)
                self.unselect()
                self.muscles[id] = m
                return

    def delete_mode(self, clicked):
        if not clicked: return
        if self.last_selected is None:
            self.switch_to_select()
            return

        id: str = self.last_selected.id
        self.last_selected = None
        part = id[0]
        if part == 'j':
            joint = self.joints[id]

            to_delete = []
            for bone_id, bone in self.bones.items():
                if bone.joint1 == joint or bone.joint2 == joint:
                    to_delete.append(bone_id)

            for bone_id in to_delete:
                self.del_bone(bone_id)

            del self.joints[id]
        if part == 'm':
            muscle = self.muscles[id]

            muscle.bone1.muscles = [m for m in muscle.bone1.muscles if m != muscle]
            muscle.bone2.muscles = [m for m in muscle.bone2.muscles if m != muscle]

            del self.muscles[id]
        if part == 'b':
            self.del_bone(id)

        self.switch_to_select()

    def del_bone(self, id):
        bone = self.bones[id]

        bone.joint1.bones = [b for b in bone.joint1.bones if b != bone]
        bone.joint2.bones = [b for b in bone.joint2.bones if b != bone]

        to_delete = []
        for muscle_id, muscle in self.muscles.items():
            if muscle.bone1 == bone or muscle.bone2 == bone:
                to_delete.append(muscle_id)

        for muscle_id in to_delete:
            del self.muscles[muscle_id]

        del self.bones[id]

    def learn(self, clicked):
        return

    def continue_func(self, clicked):
        pass

    def neural_network_button(self, clicked):
        conf = ConfigureNetwork(self.window, self.depth, self.layer_widths)
        self.depth, self.layer_widths = conf.start()

        self.switch_to_select()

    def load(self, _=None):
        def get_file_dialog(callback):
            def thread_func():
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select a JSON file",
                    filetypes=[("JSON Files", "*.json")]
                )
                root.destroy()
                callback(file_path)

            t = threading.Thread(target=thread_func)
            t.start()

        self.switch_to_select()

        def on_file_selected(file_path):
            if not file_path:
                return

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

            # Compute relative path from project root
            relative_path = os.path.relpath(file_path, project_root)

            data = CreatureLoader.load("./" + relative_path)
            if not data:
                print("Failed to load creature.")
                return
            self.clear()

            self.creature_to_ui(data)

        get_file_dialog(on_file_selected)

        self.switch_to_select()

    def switch_to_select(self):
        self.mode = 'select'
        self.mode_func = self.select_mode

    def save(self, _):
        done = CreatureLoader.save(self.joints.values(), self.bones.values(), self.muscles.values())

        self.print_notif(done)

        self.switch_to_select()

    def print_notif(self, saved):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = False

            pygame.draw.rect(self.window, colors.background_secondary, pygame.Rect(500, 360, 200, 80), border_radius=5)
            if saved:
                TextRenderer.render_text("Saved", 16, colors.foreground, (571, 390), self.window)
            else:
                TextRenderer.render_text("Couldn't Save", 16, colors.foreground, (532, 390), self.window)
            pygame.display.flip()

    def creature_to_ui(self, creature: Creature):
        for joint in creature.joints:
            pos = (joint.x, joint.y)
            id = joint.id
            j = Joint(InputHandler.coords_to_pos(pos), pos, id)
            self.joints[id] = j
            self.joint_num += 1

        for bone in creature.bones:
            id = bone.id
            b = Bone(id, self.joints[bone.joint1_id], self.joints[bone.joint2_id])
            self.bones[id] = b
            self.bone_num += 1

        for muscle in creature.muscles:
            id = muscle.id
            m = Muscle(id, self.bones[muscle.bone1_id], self.bones[muscle.bone2_id])
            self.muscles[id] = m
            self.muscle_num += 1

    def clear(self, _=None):
        self.unselect()
        self.mode_func = self.select_mode
        self.mode = "select"
        self.joints.clear()
        self.bones.clear()
        self.muscles.clear()
        self.bone_num = 0
        self.muscle_num = 0
        self.joint_num = 0
