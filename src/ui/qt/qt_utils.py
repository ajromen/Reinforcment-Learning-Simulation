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


PILL_SCROLLBAR = """
        QScrollBar:vertical {
            background: transparent;
            width: 8px;
            margin: 4px 2px 4px 2px;
        }

        QScrollBar::handle:vertical {
            background: rgba(180, 180, 180, 120);
            border-radius: 4px;
            min-height: 30px;
        }

        QScrollBar::handle:vertical:hover {
            background: rgba(200, 200, 200, 180);
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: none;
        }
        
        #H
        QScrollBar:horizontal {
            background: transparent;
            height: 8px;                 
            margin: 2px 4px 2px 4px;    
        }
        
        QScrollBar::handle:horizontal {
            background: rgba(180, 180, 180, 120);
            border-radius: 4px;
            min-width: 30px;          
        }
        
        QScrollBar::handle:horizontal:hover {
            background: rgba(200, 200, 200, 180);
        }
        
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {
            background: none;
        }
            
        """