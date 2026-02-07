from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QHBoxLayout, QFrame

from src.ui.colors import light_background, background_secondary, background_primary, foreground


class SmallButton(QFrame):
    def __init__(self, title: str, on_click):
        super().__init__()

        self.on_click = on_click

        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {background_secondary};
                border-radius: 5px;
            }}
        """)

        self.layout = QHBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(23, 0, 23, 0)

        self.title = QLabel(title)
        self.title.setStyleSheet(
            f"color: {foreground}; font-size: 16px;"
        )

        self.layout.addWidget(self.title)

    def enterEvent(self, event):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {background_primary};
                border-radius: 5px;
            }}
        """)

        self.title.setStyleSheet("color: #ffffff; font-size: 16px;")

    def leaveEvent(self, event):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {background_secondary};
                border-radius: 5px;
            }}
        """)

        self.title.setStyleSheet(
            f"color: {foreground}; font-size: 16px;"
        )

    def mousePressEvent(self, event):
        self.on_click()
