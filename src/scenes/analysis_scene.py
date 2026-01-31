import sys
from pathlib import Path

from PyQt6.QtGui import QFontDatabase, QFont
from PyQt6.QtWidgets import QFrame, QWidget, QApplication, QHBoxLayout

from src.models.creature import Creature
from src.ui.qt import qt_utils
from src.ui.qt.panel import Panel
from src.ui.qt.window import MainWindow
from src.utils.constants import ASSETS_PATH


class AnalysisScene:
    def __init__(self, creature: Creature= None):
        self.app = QApplication(sys.argv)
        qt_utils.load_font()
        self.window = MainWindow()
        self._setup_window()

    def _setup_window(self):

        layout = QHBoxLayout(self.window)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)

        left = Panel(
            "REINFORCE",
            "REINFORCE is a simple reinforcement learning method used in smaller scale",
            ASSETS_PATH + "brain_alt.png",
            ASSETS_PATH + "brain.png",
        )

        right = Panel(
            "PPO",
            "PPO or instead some other method is not important — GPT and LLMs use it",
            ASSETS_PATH + "brain_alt.png",
            ASSETS_PATH + "brain.png",
        )

        layout.addWidget(left)
        layout.addWidget(right)

    def start(self):
        self.window.show()
        self.app.exec()


