from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QHBoxLayout, QFrame

from src.ui.colors import light_background, background_secondary, background_primary


class Button(QFrame):
    def __init__(self, title: str, active_icon: str, inactive_icon: str, on_click):
        super().__init__()

        self.on_click = on_click

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(200, 80)

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {light_background};
                border-radius: 5px;
            }}
        """)

        self.layout = QHBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(23, 0, 23, 0)

        self.icon_inactive = QLabel()
        self.icon_active = QLabel()

        pixmap = QPixmap(inactive_icon)
        self.icon_inactive.setPixmap(
            pixmap.scaled(
                44, 44,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        pixmap = QPixmap(active_icon)
        self.icon_active.setPixmap(
            pixmap.scaled(
                44, 44,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self.icon_active.hide()

        self.title = QLabel(title)
        self.title.setStyleSheet(
            f"color: {background_secondary}; font-size: 16px;"
        )


        self.layout.addWidget(self.icon_inactive)
        self.layout.addWidget(self.icon_active)
        self.layout.addWidget(self.title)

    def enterEvent(self, event):
        self.icon_inactive.hide()
        self.icon_active.show()

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {background_primary};
                border-radius: 5px;
            }}
        """)

        self.title.setStyleSheet("color: #ffffff; font-size: 16px;")

    def leaveEvent(self, event):
        self.icon_active.hide()
        self.icon_inactive.show()

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {light_background};
                border-radius: 5px;
            }}
        """)

        self.title.setStyleSheet(
            f"color: {background_secondary}; font-size: 16px;"
        )

    def mousePressEvent(self, event):
        self.on_click()
