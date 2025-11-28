from src.ui.button import Button
from src.ui.image_manager import ImageManager
from src.utils.constants import WINDOW_HEIGHT, WINDOW_WIDTH, SMALL_BUTTON_DIMENSIONS, LARGE_BUTTON_DIMENSIONS
from pygame import Rect


class ButtonsManager:
    def __init__(self, window):
        self.creation_scene_buttons = []
        self.window = window

    def create_creation_scene_buttons(self):
        select = Button(10, 10, *SMALL_BUTTON_DIMENSIONS, ImageManager.select_button, 'select')
        joint = Button(10, 100, *SMALL_BUTTON_DIMENSIONS, ImageManager.joint_button, 'joint')
        bone = Button(10, 190, *SMALL_BUTTON_DIMENSIONS, ImageManager.bone_button, 'bone')
        muscle = Button(10, 280, *SMALL_BUTTON_DIMENSIONS, ImageManager.muscle_button, 'muscle')

        delete = Button(10, WINDOW_HEIGHT - 180, *SMALL_BUTTON_DIMENSIONS, ImageManager.delete_button, 'delete')
        clear = Button(10, WINDOW_HEIGHT - 90, *SMALL_BUTTON_DIMENSIONS, ImageManager.clear_button, 'clear')

        neural_network = Button(WINDOW_WIDTH - 90, 10, *SMALL_BUTTON_DIMENSIONS, ImageManager.neural_network_button,
                                'neural_network')
        evolve = Button(WINDOW_WIDTH - 210, WINDOW_HEIGHT - 90, *LARGE_BUTTON_DIMENSIONS, ImageManager.evolve_button,
                        'evolve')

        self.creation_scene_buttons = [select, muscle, joint, bone, delete, clear, neural_network, evolve]

    def show_check_creation_scene_buttons(self, clicked):
        ret = None
        for button in self.creation_scene_buttons:
            action = button.check_show(self.window)
            if clicked and action:
                ret = button.name
                clicked = False
        return ret
