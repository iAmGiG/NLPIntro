from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QHBoxLayout, QScrollArea, QSpinBox
)
from PyQt6.QtCore import pyqtSlot


def display_error(self, message: str, field: QLineEdit):
    field.setStyleSheet("border: 1px solid red;")
    error_label = QLabel(message)
    error_label.setStyleSheet("color: red;")
    self.layout.addWidget(error_label)


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

        # Training parameters layout
        self.training_params_layout = QVBoxLayout()
        self.iterations_label = QLabel("Number of Iterations:")
        self.iterations_input = QSpinBox()
        self.iterations_input.setMinimum(1)
        self.iterations_input.setValue(2)  # Default value
        self.training_params_layout.addWidget(self.iterations_label)
        self.training_params_layout.addWidget(self.iterations_input)

        # Combine input fields and training params
        self.combined_input_layout = QHBoxLayout()
        self.combined_input_layout.addLayout(self.input_layout)
        self.combined_input_layout.addLayout(self.training_params_layout)

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

        # Add to layout
        self.layout.addLayout(self.combined_input_layout)
        self.layout.addLayout(self.buttons_layout)
        self.layout.addWidget(self.latex_label)
        self.layout.addWidget(self.latex_preview)

        self.setLayout(self.layout)
