import sys
from multiprocessing import Process, Queue

import pygame
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QHBoxLayout

from src.agents.ppo_agent import  PPOAgent
from src.agents.reinforce_agent import ReinforceAgent
from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.simulation.simulation_window import SimulationWindow
from src.ui.qt import qt_utils
from src.ui.qt.panel import Panel
from src.ui.qt.window import MainWindow
from src.utils.constants import ASSETS_PATH, SAVE_FILE_PATH

class AnalysisScene:
    def __init__(self, creature: Creature, nn_layers: list[int]):
        self.creature = creature
        self.nn_layers = nn_layers
        self.app = QApplication(sys.argv)
        qt_utils.load_font()
        self.window = MainWindow()
        self._setup_window()
        self.files_root = SAVE_FILE_PATH  # + creature.id
        self.queue = Queue()
        input_size = CreaturePymunk.get_number_of_inputs(creature)
        self.nn_layers[-1] = len(creature.muscles)
        self.nn_layers[0] = input_size
        print(nn_layers)

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
        self.left.button.setEnabled(False)
        self.right.button.setEnabled(False)
        p = Process(target=self._reinforce_process, args=(self.queue,))

        p.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self._check_queue)
        self.timer.start(5000)  # proverava svakih 5s

    def run_ppo(self):
        self.left.button.setEnabled(False)
        self.right.button.setEnabled(False)
        p = Process(target=self._ppo_process, args=(self.queue,))

        p.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self._check_queue)
        self.timer.start(5000)  # proverava svakih 5s

    def _reinforce_process(self, queue):
        agent = ReinforceAgent(self.nn_layers)
        reinforce_window = SimulationWindow(self.creature, agent)
        reinforce_window.start()
        pygame.quit()
        queue.put("reinforce")

    def _ppo_process(self, queue):
        agent = PPOAgent(self.nn_layers)
        reinforce_window = SimulationWindow(self.creature, agent)
        reinforce_window.start()
        pygame.quit()
        queue.put("ppo")

    def _check_queue(self):
        if not self.queue.empty():
            self.timer.stop()
            process = self.queue.get()
            if process == "reinforce":
                self.reinforce_finished()
            elif process == "ppo":
                self.ppo_finished()

    def ppo_finished(self):
        self.left.button.setEnabled(True)
        self.right.button.setEnabled(False)
        self.right.load_markdown()

    def reinforce_finished(self):
        self.left.button.setEnabled(False)
        self.right.button.setEnabled(True)
        self.left.load_markdown()
