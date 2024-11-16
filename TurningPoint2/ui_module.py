from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QHBoxLayout, QSpinBox
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

        # Initialize scores
        self.fluency_score = 0.0
        self.adequacy_score = 0.0
        self.bleu_score = 0.0

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

        # Iterations input
        self.iterations_label = QLabel("Number of Iterations:")
        self.iterations_input = QSpinBox()
        self.iterations_input.setMinimum(1)
        self.iterations_input.setValue(2)  # Default value

        self.input_layout.addWidget(self.source_label)
        self.input_layout.addWidget(self.source_input)
        self.input_layout.addWidget(self.target_label)
        self.input_layout.addWidget(self.target_input)
        self.input_layout.addWidget(self.iterations_label)
        self.input_layout.addWidget(self.iterations_input)

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
        self.latex_preview.setMinimumHeight(200)

        # Scores display
        self.scores_layout = QVBoxLayout()
        self.fluency_label = QLabel("Fluency Score: N/A")
        self.adequacy_label = QLabel("Adequacy Score: N/A")
        self.bleu_label = QLabel("BLEU Score: N/A")

        self.scores_layout.addWidget(self.fluency_label)
        self.scores_layout.addWidget(self.adequacy_label)
        self.scores_layout.addWidget(self.bleu_label)

        # Error message label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")

        # Add widgets to main layout
        self.layout.addLayout(self.input_layout)
        self.layout.addLayout(self.buttons_layout)
        self.layout.addWidget(self.error_label)
        self.layout.addLayout(self.scores_layout)
        self.layout.addWidget(self.latex_label)
        self.layout.addWidget(self.latex_preview)

        self.setLayout(self.layout)

    def update_scores_display(self, fluency_score: float, adequacy_score: float, bleu_score: float):
        self.fluency_label.setText(f"Fluency Score: {fluency_score:.4f}")
        self.adequacy_label.setText(f"Adequacy Score: {adequacy_score:.4f}")
        self.bleu_label.setText(f"BLEU Score: {bleu_score:.4f}")
