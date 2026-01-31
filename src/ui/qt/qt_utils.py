from PyQt6.QtGui import QFontDatabase, QFont
from PyQt6.QtWidgets import QApplication


def load_font():
    font_id = QFontDatabase.addApplicationFont(
        str("assets/JetBrainsMono-Regular.ttf")
    )
    if font_id == -1:
        return

    family = QFontDatabase.applicationFontFamilies(font_id)[0]
    QApplication.setFont(QFont(family))