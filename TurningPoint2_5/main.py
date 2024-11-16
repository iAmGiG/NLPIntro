from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel)
from PyQt6.QtCore import Qt
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Translation Trainer")
        self.setMinimumSize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Training pair inputs
        self.create_input_section(layout)

        # Output section
        self.create_output_section(layout)

        # Control buttons
        self.create_control_section(layout)

    def create_input_section(self, parent_layout):
        # First training pair
        pair1_group = QWidget()
        pair1_layout = QVBoxLayout(pair1_group)

        pair1_layout.addWidget(QLabel("Training Pair 1:"))
        self.source1 = QLineEdit()
        self.target1 = QLineEdit()
        self.source1.setPlaceholderText("Enter source sentence (English)")
        self.target1.setPlaceholderText("Enter target sentence (Spanish)")
        pair1_layout.addWidget(self.source1)
        pair1_layout.addWidget(self.target1)

        # Second training pair
        pair2_group = QWidget()
        pair2_layout = QVBoxLayout(pair2_group)

        pair2_layout.addWidget(QLabel("Training Pair 2:"))
        self.source2 = QLineEdit()
        self.target2 = QLineEdit()
        self.source2.setPlaceholderText("Enter source sentence (English)")
        self.target2.setPlaceholderText("Enter target sentence (Spanish)")
        pair2_layout.addWidget(self.source2)
        pair2_layout.addWidget(self.target2)

        parent_layout.addWidget(pair1_group)
        parent_layout.addWidget(pair2_group)

    def create_output_section(self, parent_layout):
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(300)
        parent_layout.addWidget(QLabel("Training Steps Output:"))
        parent_layout.addWidget(self.output_text)

    def create_control_section(self, parent_layout):
        control_layout = QHBoxLayout()

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.on_train)

        self.export_button = QPushButton("Export to LaTeX")
        self.export_button.clicked.connect(self.on_export)

        control_layout.addWidget(self.train_button)
        control_layout.addWidget(self.export_button)

        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        parent_layout.addWidget(control_widget)

    def on_train(self):
        # Get training pairs
        pair1_source = self.source1.text().strip().split()
        pair1_target = self.target1.text().strip().split()
        pair2_source = self.source2.text().strip().split()
        pair2_target = self.target2.text().strip().split()

        # Validate inputs
        if not all([pair1_source, pair1_target, pair2_source, pair2_target]):
            self.output_text.setText("Please enter both training pairs")
            return

        # Create training pairs
        from src.models.ibm_model1 import TranslationPair, IBMModel1
        training_pairs = [
            TranslationPair(pair1_source, pair1_target),
            TranslationPair(pair2_source, pair2_target)
        ]

        # Train model and show steps
        model = IBMModel1()
        history = model.train(training_pairs, num_iterations=2)

        # Display results
        self.show_training_steps(history)

    def show_training_steps(self, history):
        # Implementation for displaying training steps
        pass

    def on_export(self):
        # Implementation for LaTeX export
        pass
