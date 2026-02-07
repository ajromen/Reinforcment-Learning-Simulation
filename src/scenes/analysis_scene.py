import sys
from multiprocessing import Process, Queue

import pygame
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QHBoxLayout
from contourpy.util.data import simple

from src.agents.ppo_agent import PPOAgent
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

        self.save_path = SAVE_FILE_PATH + str(creature.id) + "/"

    def _setup_window(self):
        layout = QHBoxLayout(self.window)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)

        self.left = Panel(
            "REINFORCE",
            "REINFORCE is a simple reinforcement learning method used in smaller scale. Updates actions based on total rewards collected in an episode.",
            ASSETS_PATH + "brain_alt.png",
            ASSETS_PATH + "brain.png",
            self.run_reinforce,
            lambda: self.run_reinforce(True)
        )

        self.right = Panel(
            "PPO",
            "PPO or proximal policy optimization is a more stable method. Uses an additional critic model to improve efficiency.",
            ASSETS_PATH + "brain_alt.png",
            ASSETS_PATH + "brain.png",
            self.run_ppo,
            lambda: self.run_ppo(True)
        )

        layout.addWidget(self.left)
        layout.addWidget(self.right)

    def start(self):
        self.window.show()
        self.app.exec()

    def run_reinforce(self, simple=False):
        p = Process(target=self._reinforce_process, args=(self.queue, simple,))
        p.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self._check_queue)
        self.timer.start(1000)

        self.left.button.setEnabled(False)
        self.right.button.setEnabled(False)

    def run_ppo(self, simple=False):
        p = Process(target=self._ppo_process, args=(self.queue, simple,))
        p.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self._check_queue)
        self.timer.start(1000)

        self.left.button.setEnabled(False)
        self.right.button.setEnabled(False)

    def _reinforce_process(self, queue, simple=False):
        agent = ReinforceAgent(self.nn_layers)
        reinforce_window = SimulationWindow(self.creature, agent, self.save_path + "reinforce/")
        if not simple:
            reinforce_window.start()
        else:
            reinforce_window.start_simple()
        pygame.quit()
        if not simple:
            queue.put("reinforce")
        else:
            queue.put("reinforce_simple")

    def _ppo_process(self, queue, simple=False):
        agent = PPOAgent(self.nn_layers)
        ppo_window = SimulationWindow(self.creature, agent, self.save_path + "ppo/")
        if not simple:
            ppo_window.start()
        else:
            ppo_window.start_simple()
        pygame.quit()
        if not simple:
            queue.put("ppo")
        else:
            queue.put("ppo_simple")

    def _check_queue(self):
        if not self.queue.empty():
            self.timer.stop()
            process = self.queue.get()
            if process == "reinforce":
                self.reinforce_finished()
            elif process == "ppo":
                self.ppo_finished()
            elif process == "ppo_simple":
                self.left.button.setEnabled(True)
                self.right.button.setEnabled(False)
            elif process == "reinforce_simple":
                self.left.button.setEnabled(False)
                self.right.button.setEnabled(True)

    def ppo_finished(self):
        self.left.button.setEnabled(True)
        self.right.button.setEnabled(False)
        self.right.load_markdown()

    def reinforce_finished(self):
        self.left.button.setEnabled(False)
        self.right.button.setEnabled(True)
        self.left.load_markdown()
