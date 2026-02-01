import os
from pathlib import Path

import markdown
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser

from src.ui.colors import light_background
from src.ui.qt.button import Button

test_md = "./tmp/README.md"


class Panel(QWidget):
    def __init__(self, title: str, description: str, active_icon: str, inactive_icon: str, on_click):
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.setSpacing(60)

        self.button = Button(title, active_icon, inactive_icon, on_click)

        self.description = QLabel(description)
        self.description.setWordWrap(True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setFixedWidth(400)
        self.description.setStyleSheet(f"color: {light_background};font-size: 14px;")

        self.layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.description, 0, Qt.AlignmentFlag.AlignHCenter)

    def load_markdown(self):
        self.button.hide()
        self.description.hide()

        viewer = QTextBrowser()


        if not os.path.exists(test_md):
            viewer.setPlainText(f"File not found:\n{test_md}")
            self.layout.addWidget(viewer)
            return


        html = markdown.markdown(
            open(test_md).read(),
            extensions=[
                "fenced_code",
                "tables",
            ]
        )

        viewer.setHtml(html)
        viewer.setOpenExternalLinks(True)

        viewer.setStyleSheet("""
            QTextBrowser {
                background-color: #0c0c0c;
                color: #e0e0e0;
                font-family: "JetBrains Mono";
                font-size: 15px;
                padding: 12px;
            }
        """)



        self.layout.addWidget(viewer)
