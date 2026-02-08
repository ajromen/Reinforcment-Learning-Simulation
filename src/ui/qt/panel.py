import os
import webbrowser
from pathlib import Path

import markdown
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser, QHBoxLayout

from src.ui.colors import light_background, background_secondary, foreground
from src.ui.qt.button import Button
from src.ui.qt.small_button import SmallButton



class Panel(QWidget):
    def __init__(self, title: str, description: str, active_icon: str, inactive_icon: str, on_click, run_simple, continue_fn):
        super().__init__()

        self.viewer = None
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.setSpacing(24)
        self.run_simple = run_simple

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(12)
        toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.button = Button(title, active_icon, inactive_icon, on_click)
        self.filepath = ""

        self.open_external_btn = SmallButton(
            "Open in browser",
            self._open_in_browser
        )

        self.run_live_btn = SmallButton(
            "Run simulation",
            self._run_live_simulation
        )

        self.continue_btn = SmallButton(
            "Continue learning",
            continue_fn
        )
        self.restart_btn = SmallButton(
            "Restart learning",
            on_click
        )

        self.run_live_btn.hide()
        self.open_external_btn.hide()
        self.continue_btn.hide()
        self.restart_btn.hide()

        toolbar_layout.addWidget(self.open_external_btn)
        toolbar_layout.addWidget(self.run_live_btn)
        toolbar_layout.addWidget(self.continue_btn)
        toolbar_layout.addWidget(self.restart_btn)


        self.description = QLabel(description)
        self.description.setWordWrap(True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setFixedWidth(400)
        self.description.setStyleSheet(f"color: {light_background};font-size: 14px;")

        self.layout.addWidget(toolbar)
        self.layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignHCenter)
        # self.layout.addSpacing(40)
        self.layout.addWidget(self.description, 0, Qt.AlignmentFlag.AlignHCenter)

    def _open_in_browser(self):
        webbrowser.open(Path(self.filepath).resolve().as_uri())

    def _run_live_simulation(self):
        self.run_simple()

    def load_markdown(self, filepath):
        self.button.hide()
        self.description.hide()
        self.run_live_btn.show()
        self.open_external_btn.show()
        self.continue_btn.show()
        self.restart_btn.show()
        if self.viewer is not None:
            self.layout.removeWidget(self.viewer)
            self.viewer = None

        self.viewer = QTextBrowser()
        self.filepath = filepath


        html = markdown.markdown(
            open(filepath).read(),
            extensions=[
                "fenced_code",
                "tables",

            ]
        )

        md_path = Path(filepath).resolve()
        base_url = QUrl.fromLocalFile(str(md_path.parent) + "/")

        self.viewer.document().setBaseUrl(base_url)
        self.viewer.setHtml(html)

        self.viewer.setOpenExternalLinks(True)

        self.viewer.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {background_secondary};
                color: {foreground};
                font-family: "JetBrains Mono";
                font-size: 15px;
                padding: 12px;
            }}
        """)

        self.layout.addWidget(self.viewer)
