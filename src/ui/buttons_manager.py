from src.ui.components.button import Button
from src.ui.image_manager import ImageManager
from src.ui.ui_settings import WINDOW_HEIGHT, WINDOW_WIDTH, SMALL_BUTTON_DIMENSIONS, LARGE_BUTTON_DIMENSIONS


class ButtonsManager:
    def __init__(self, window):
        self.creation_scene_buttons = []
        self.window = window
        self.continue_btn_visible = False

    def create_creation_scene_buttons(self):
        select = Button(10, 10, *SMALL_BUTTON_DIMENSIONS, ImageManager.select_button,
                        ImageManager.select_button_inactive, 'select')
        joint = Button(10, 100, *SMALL_BUTTON_DIMENSIONS, ImageManager.joint_button, ImageManager.joint_button_inactive,
                       'joint')
        bone = Button(10, 190, *SMALL_BUTTON_DIMENSIONS, ImageManager.bone_button, ImageManager.bone_button_inactive,
                      'bone')
        muscle = Button(10, 280, *SMALL_BUTTON_DIMENSIONS, ImageManager.muscle_button,
                        ImageManager.muscle_button_inactive, 'muscle')

        delete = Button(10, WINDOW_HEIGHT - 180, *SMALL_BUTTON_DIMENSIONS, ImageManager.delete_button,
                        ImageManager.delete_button_inactive, 'delete')
        clear = Button(10, WINDOW_HEIGHT - 90, *SMALL_BUTTON_DIMENSIONS, ImageManager.clear_button,
                       ImageManager.clear_button_inactive, 'clear')

        neural_network = Button(WINDOW_WIDTH - 90, 10, *SMALL_BUTTON_DIMENSIONS, ImageManager.neural_network_button,
                                ImageManager.neural_network_button_inactive,
                                'neural_network')
        learn = Button(WINDOW_WIDTH - 210, WINDOW_HEIGHT - 90, *LARGE_BUTTON_DIMENSIONS, ImageManager.learn_button,
                        ImageManager.learn_button_inactive,
                        'learn')

        load = Button(WINDOW_WIDTH - 90, WINDOW_HEIGHT - 180, *SMALL_BUTTON_DIMENSIONS, ImageManager.load_button,
                        ImageManager.load_button_inactive,
                        'load')

        save = Button(WINDOW_WIDTH - 180, WINDOW_HEIGHT - 180, *SMALL_BUTTON_DIMENSIONS, ImageManager.save_button,
                      ImageManager.save_button_inactive,
                      'save')

        self.continue_btn = Button(WINDOW_WIDTH - 418, WINDOW_HEIGHT - 90, *LARGE_BUTTON_DIMENSIONS, ImageManager.continue_button,
                      ImageManager.continue_button_inactive,
                      'continue')

        self.creation_scene_buttons = [select, muscle, joint, bone, delete, clear, neural_network, learn, load, save]

    def show_check_creation_scene_buttons(self, clicked, mode):
        ret = None
        for button in self.creation_scene_buttons:
            if button.name == mode:
                button.active = True
            else:
                button.active = False
            action = button.check_show(self.window)
            if clicked and action:
                ret = button.name
                clicked = False
        return ret

    def show_continue_button(self):
        self.creation_scene_buttons.append(self.continue_btn)
        self.continue_btn_visible = True

    def hide_continue_button(self):
        if self.continue_btn_visible:
            self.creation_scene_buttons.remove(self.continue_btn)
            self.continue_btn_visible = False
