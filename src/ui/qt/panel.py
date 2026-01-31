from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser

from src.ui.colors import light_background
from src.ui.qt.button import Button

test_md = Path("tmp/README.md")

class Panel(QWidget):
    def __init__(self, title: str, description: str, active_icon: str, inactive_icon: str):
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.setSpacing(60)

        self.button = Button(title, active_icon, inactive_icon, self.load_markdown)

        self.description = QLabel(description)
        self.description.setWordWrap(True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setFixedWidth(400)
        self.description.setStyleSheet(f"color: {light_background};font-size: 14px;")

        self.layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.description, 0, Qt.AlignmentFlag.AlignHCenter)

    def load_markdown(self):
        self.button.deleteLater()
        self.description.deleteLater()

        viewer = QTextBrowser()
        viewer.setStyleSheet(f"""
            QTextBrowser {{
                background-color: transparent;
                color: #e0e0e0;
                font-size: 16px;
            }}
        """)

        if test_md.exists():
            viewer.setMarkdown(test_md.read_text(encoding="utf-8"))
        else:
            viewer.setPlainText(f"File not found:\n{test_md}")

        self.layout.addWidget(viewer)

