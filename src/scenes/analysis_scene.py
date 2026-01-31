import sys
from multiprocessing import Process

import pygame
from PyQt6.QtWidgets import QApplication, QHBoxLayout

from src.models.creature import Creature
from src.simulation.simulation_window import SimulationWindow
from src.ui.qt import qt_utils
from src.ui.qt.panel import Panel
from src.ui.qt.window import MainWindow
from src.utils.constants import ASSETS_PATH


class AnalysisScene:
    def __init__(self, creature: Creature, nn_layers: list[int]):
        self.creature = creature
        self.nn_layers = nn_layers
        self.app = QApplication(sys.argv)
        qt_utils.load_font()
        self.window = MainWindow()
        self.left = None
        self.right = None
        self._setup_window()
        self.files_root = "./data/"  # + creature.id

    def _setup_window(self):
        layout = QHBoxLayout(self.window)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)

        self.left = Panel(
            "REINFORCE",
            "REINFORCE is a simple reinforcement learning method used in smaller scale. Updates actions based on total rewards collected in an episode.",
            ASSETS_PATH + "brain_alt.png",
            ASSETS_PATH + "brain.png",
            self.run_reinforce
        )

        self.right = Panel(
            "PPO",
            "PPO or proximal policy optimization is a more stable method. Uses an additional critic model to improve efficiency.",
            ASSETS_PATH + "brain_alt.png",
            ASSETS_PATH + "brain.png",
            self.run_ppo
        )

        layout.addWidget(self.left)
        layout.addWidget(self.right)

    def start(self):
        self.window.show()
        self.app.exec()

    def run_reinforce(self):
        callback = self.reinforce_finished

        def simulation_process(callback):
            reinforce_window = SimulationWindow(self.creature, None)
            reinforce_window.start()
            pygame.quit()
            # ne moze ovako moras da dodas QTimer i Queue iz multiprocessinga
            callback()

        p = Process(target=simulation_process,args=(callback,))
        p.start()

    def run_ppo(self):
        def simulation_process():
            reinforce_window = SimulationWindow(self.creature, None)
            reinforce_window.start()
            pygame.quit()
            self.ppo_finished()

        p = Process(target=simulation_process)
        p.start()

    def ppo_finished(self):
        self.right.load_markdown()

    def reinforce_finished(self):
        self.left.load_markdown()