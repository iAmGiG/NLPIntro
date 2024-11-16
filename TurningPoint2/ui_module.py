# ui_module.py
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QHBoxLayout, QScrollArea
)
from PyQt6.QtCore import pyqtSlot


class TranslationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Translation Tool")

        # Main layout
        self.layout = QVBoxLayout()

        # Input fields layout
        self.input_layout = QVBoxLayout()
        self.source_label = QLabel("Source Sentence (English):")
        self.source_input = QLineEdit()
        self.source_input.setText("the house is small")
        self.target_label = QLabel("Target Sentence (Spanish):")
        self.target_input = QLineEdit()
        self.target_input.setText("la casa es pequeña")

        self.input_layout.addWidget(self.source_label)
        self.input_layout.addWidget(self.source_input)
        self.input_layout.addWidget(self.target_label)
        self.input_layout.addWidget(self.target_input)

        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        self.train_button = QPushButton("Train Model")
        self.save_button = QPushButton("Save to .tex")
        self.save_button.setEnabled(False)  # Disabled until training is done

        self.buttons_layout.addWidget(self.train_button)
        self.buttons_layout.addWidget(self.save_button)

        # LaTeX preview area
        self.latex_label = QLabel("LaTeX Output Preview:")
        self.latex_preview = QTextEdit()
        self.latex_preview.setReadOnly(True)
        self.latex_preview.setAcceptRichText(False)

        # Add widgets to main layout
        self.layout.addLayout(self.input_layout)
        self.layout.addLayout(self.buttons_layout)
        self.layout.addWidget(self.latex_label)
        self.layout.addWidget(self.latex_preview)

        self.setLayout(self.layout)
