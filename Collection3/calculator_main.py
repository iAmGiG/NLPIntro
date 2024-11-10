import sys
from PyQt6.QtWidgets import QApplication
from tfidf_window import TFIDFWindow


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    window = TFIDFWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
