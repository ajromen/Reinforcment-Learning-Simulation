import os
import webbrowser
from pathlib import Path

import markdown
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser, QHBoxLayout

from src.ui.colors import light_background, background_secondary, foreground
from src.ui.qt.button import Button
from src.ui.qt.small_button import SmallButton

test_md = "./tmp/README.md"


class Panel(QWidget):
    def __init__(self, title: str, description: str, active_icon: str, inactive_icon: str, on_click, run_simple):
        super().__init__()

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

        self.open_external_btn = SmallButton(
            "Open in browser",
            self._open_in_browser
        )

        self.run_live_btn = SmallButton(
            "Run simulation",
            self._run_live_simulation
        )

        self.run_live_btn.hide()
        self.open_external_btn.hide()

        toolbar_layout.addWidget(self.open_external_btn)
        toolbar_layout.addWidget(self.run_live_btn)


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
        webbrowser.open(Path(test_md).resolve().as_uri())

    def _run_live_simulation(self):
        self.run_simple()

    def load_markdown(self):
        self.button.hide()
        self.description.hide()
        self.run_live_btn.show()
        self.open_external_btn.show()

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

        md_path = Path(test_md).resolve()
        base_url = QUrl.fromLocalFile(str(md_path.parent) + "/")

        viewer.document().setBaseUrl(base_url)
        viewer.setHtml(html)

        viewer.setOpenExternalLinks(True)

        viewer.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {background_secondary};
                color: {foreground};
                font-family: "JetBrains Mono";
                font-size: 15px;
                padding: 12px;
            }}
        """)

        self.layout.addWidget(viewer)
