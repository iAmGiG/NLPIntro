from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QSpinBox, QPushButton,
                             QTextEdit, QScrollArea, QGridLayout)
from ibm_model1 import IBMModel1


class IBMModel1LatexGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IBM Model 1 LaTeX Generator")
        self.setMinimumSize(800, 600)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Input section
        input_group = QWidget()
        input_layout = QGridLayout(input_group)

        # English sentence input
        input_layout.addWidget(QLabel("English sentence:"), 0, 0)
        self.eng_input = QLineEdit()
        self.eng_input.setText("the house is small")  # Example training pair
        input_layout.addWidget(self.eng_input, 0, 1)

        # Spanish sentence input
        input_layout.addWidget(QLabel("Spanish sentence:"), 1, 0)
        self.foreign_input = QLineEdit()
        self.foreign_input.setText(
            "la casa es pequeña")  # Example training pair
        input_layout.addWidget(self.foreign_input, 1, 1)

        # Iteration count
        input_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setRange(1, 10)
        self.iter_spinbox.setValue(2)  # Set default to 2 iterations
        input_layout.addWidget(self.iter_spinbox, 2, 1)

        layout.addWidget(input_group)

        # Generate button
        self.generate_btn = QPushButton("Generate LaTeX")
        self.generate_btn.clicked.connect(self.generate_latex)
        layout.addWidget(self.generate_btn)

        # Output section
        output_scroll = QScrollArea()
        output_scroll.setWidgetResizable(True)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        output_scroll.setWidget(self.output_text)
        layout.addWidget(output_scroll)

    def generate_latex(self):
        eng_sentence = self.eng_input.text()
        foreign_sentence = self.foreign_input.text()
        iterations = self.iter_spinbox.value()

        # Create and train IBM Model 1
        model = IBMModel1(eng_sentence, foreign_sentence)
        convergence_history, iteration_tables = model.train(iterations)

        # Debugging: Print the contents of iteration_tables
        print(f"Total Iterations: {iterations}")
        for i, table in enumerate(iteration_tables):
            print(f"Iteration {i + 1}: {table}")

        # Generate complete LaTeX document
        latex = self.generate_latex_document(
            model, convergence_history, iteration_tables)
        self.output_text.setText(latex)

    def generate_latex_document(self, model, convergence_history, iteration_tables):
        latex = []

        # Document preamble
        latex.extend([
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{array}",
            "\\usepackage{multirow}",
            "\\usepackage{float}",
            "\\usepackage[table]{xcolor}",
            "\\usepackage{amsmath}",
            "\\usepackage{tikz}",
            "\\begin{document}",
            "",
            "\\section{IBM Model 1 Training Results}",
            "",
            f"\\textbf{{Training Pair:}}\\\\",
            f"English: {' '.join(model.eng_sentence)}\\\\",
            f"Spanish: {' '.join(model.foreign_sentence)}\\\\",
            ""
        ])

        # Add convergence table
        latex.append("\\subsection{Convergence of Translation Probabilities}")
        latex.append(model.generate_convergence_table())
        latex.append("")

        # Add iteration tables with perplexity
        latex.append("\\subsection{Detailed Iteration Results}")
        if not iteration_tables:
            latex.append("\\textbf{No iterations data available.}")
        else:
            for i, iteration_data in enumerate(iteration_tables):
                latex.append(f"\\subsubsection{{Iteration {i + 1}}}")
                latex.append(model.generate_latex_table(iteration_data))
                latex.append("")

        # Close document
        latex.append("\\end{document}")

        return "\n".join(latex)
