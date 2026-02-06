from PyQt6.QtWidgets import QWidget

from src.ui.colors import foreground, background_secondary
from src.ui.qt.qt_utils import PILL_SCROLLBAR
from src.ui.ui_settings import APP_NAME


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.resize(1200, 800)

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {background_secondary};
            }}
        """+PILL_SCROLLBAR)
