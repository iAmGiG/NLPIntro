import sys
from PyQt6.QtWidgets import QApplication
from gui import IBMModel1LatexGenerator


def main():
    app = QApplication(sys.argv)
    window = IBMModel1LatexGenerator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
