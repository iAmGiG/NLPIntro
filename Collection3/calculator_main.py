import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QPushButton, QLabel,
                             QFileDialog, QGridLayout)
from PyQt6.QtGui import QClipboard
from tfidf_calculator import TFIDFCalculator
from tfidf_formatter import TFIDFFormatter
from relevance_scorer import RelevanceScorer


class TFIDFWindow(QMainWindow):
    """Main window for the TF-IDF calculator application"""

    def __init__(self):
        super().__init__()
        self.calculator = TFIDFCalculator()
        self.initUI()

    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('TF-IDF Calculator')
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create input section
        input_layout = QGridLayout()

        # Query input
        input_layout.addWidget(QLabel("Query:"), 0, 0)
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter query here...")
        self.query_input.setMaximumHeight(50)
        input_layout.addWidget(self.query_input, 0, 1)

        # Answer inputs
        for i in range(3):
            input_layout.addWidget(QLabel(f"Answer {i+1}:"), i+1, 0)
            answer_input = QTextEdit()
            answer_input.setPlaceholderText(f"Enter answer {i+1} here...")
            answer_input.setMaximumHeight(50)
            setattr(self, f"answer{i+1}_input", answer_input)
            input_layout.addWidget(answer_input, i+1, 1)

        layout.addLayout(input_layout)

        # Buttons
        button_layout = QHBoxLayout()

        calculate_btn = QPushButton("Calculate TF-IDF")
        calculate_btn.clicked.connect(self.calculate_tfidf)
        button_layout.addWidget(calculate_btn)

        save_latex_btn = QPushButton("Save LaTeX")
        save_latex_btn.clicked.connect(self.save_latex)
        button_layout.addWidget(save_latex_btn)

        layout.addLayout(button_layout)

        # Output sections
        output_layout = QHBoxLayout()

        # LaTeX output
        latex_group = QVBoxLayout()
        latex_group.addWidget(QLabel("LaTeX Output:"))
        self.latex_output = QTextEdit()
        self.latex_output.setReadOnly(True)
        latex_group.addWidget(self.latex_output)

        # Add Copy Button for LaTeX Output
        copy_latex_btn = QPushButton("Copy LaTeX to Clipboard")
        copy_latex_btn.clicked.connect(self.copy_latex_to_clipboard)
        latex_group.addWidget(copy_latex_btn)

        output_layout.addLayout(latex_group)

        # Human readable output
        human_group = QVBoxLayout()
        human_group.addWidget(QLabel("Human Readable Output:"))
        self.human_output = QTextEdit()
        self.human_output.setReadOnly(True)
        human_group.addWidget(self.human_output)

        # Add Copy Button for Human Readable Output
        copy_human_btn = QPushButton("Copy Human Readable to Clipboard")
        copy_human_btn.clicked.connect(self.copy_human_to_clipboard)
        human_group.addWidget(copy_human_btn)

        output_layout.addLayout(human_group)

        layout.addLayout(output_layout)

        # Set default example texts
        self.query_input.setText("What causes global warming?")
        self.answer1_input.setText(
            "Global warming is caused by an increase in greenhouse gases.")
        self.answer2_input.setText(
            "Deforestation contributes to global warming.")
        self.answer3_input.setText(
            "Electric vehicles help reduce carbon emissions.")

        # Part B Output Sections
        partb_output_layout = QHBoxLayout()

        # Part B Human Readable Output
        partb_human_group = QVBoxLayout()
        partb_human_group.addWidget(QLabel("Part B Human Readable Output:"))
        self.partb_human_output = QTextEdit()
        self.partb_human_output.setReadOnly(True)
        partb_human_group.addWidget(self.partb_human_output)

        # Add Copy Button for Part B Human Readable Output
        copy_partb_human_btn = QPushButton(
            "Copy Part B Human Readable to Clipboard")
        copy_partb_human_btn.clicked.connect(
            self.copy_partb_human_to_clipboard)
        partb_human_group.addWidget(copy_partb_human_btn)

        partb_output_layout.addLayout(partb_human_group)

        # Part B LaTeX Output
        partb_latex_group = QVBoxLayout()
        partb_latex_group.addWidget(QLabel("Part B LaTeX Output:"))
        self.partb_latex_output = QTextEdit()
        self.partb_latex_output.setReadOnly(True)
        partb_latex_group.addWidget(self.partb_latex_output)

        # Add Copy Button for Part B LaTeX Output
        copy_partb_latex_btn = QPushButton("Copy Part B LaTeX to Clipboard")
        copy_partb_latex_btn.clicked.connect(
            self.copy_partb_latex_to_clipboard)
        partb_latex_group.addWidget(copy_partb_latex_btn)

        partb_output_layout.addLayout(partb_latex_group)

        layout.addLayout(partb_output_layout)

    def calculate_tfidf(self):
        """Calculate TF-IDF and update outputs."""
        query = self.query_input.toPlainText()
        # Correctly access the individual answer inputs
        answers = [
            self.answer1_input.toPlainText(),
            self.answer2_input.toPlainText(),
            self.answer3_input.toPlainText()
        ]

        texts = [query] + answers
        self.calculator.text_labels = ["Q", "A1", "A2", "A3"]

        # Perform the TF-IDF calculation
        self.calculator.compute_tfidf(texts)

        # Get the vocabulary and TF-IDF vectors from the calculator
        vocabulary = self.calculator.vocabulary
        tfidf_vectors = self.calculator.tfidf_vectors

        # Update the outputs using the formatter
        self.latex_output.setText(
            TFIDFFormatter.generate_latex(
                vocabulary=vocabulary,
                tfidf_vectors=tfidf_vectors,
                text_labels=self.calculator.text_labels
            )
        )
        self.human_output.setText(
            TFIDFFormatter.generate_human_readable(
                vocabulary=vocabulary,
                tfidf_vectors=tfidf_vectors,
                text_labels=self.calculator.text_labels
            )
        )

    def save_latex(self):
        """Save LaTeX output to file"""
        if self.calculator.tfidf_vectors is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save LaTeX File",
            "",
            "LaTeX Files (*.tex);;All Files (*)"
        )

        if file_name:
            if not file_name.endswith('.tex'):
                file_name += '.tex'

            latex_content = (
                "\\documentclass[12pt]{article}\n"
                "\\usepackage{amsmath}\n"
                "\\usepackage{geometry}\n"
                "\\geometry{margin=1in}\n"
                "\\begin{document}\n\n"
                "% TF-IDF Matrix for Query and Answers\n"
                "\\[\n"
                f"{self.calculator.generate_latex()}\n"
                "\\]\n\n"
                "\\end{document}"
            )

            try:
                with open(file_name, 'w') as f:
                    f.write(latex_content)
                print(f"Successfully saved LaTeX file to {file_name}")
            except Exception as e:
                print(f"Error saving file: {e}")

    def calculate_relevance_scores(self):
        """Calculate relevance scores and update Part B outputs."""
        query = self.query_input.toPlainText()
        answers = [
            self.answer1_input.toPlainText(),
            self.answer2_input.toPlainText(),
            self.answer3_input.toPlainText()
        ]

        # Ensure Part A calculations are done
        if self.calculator.tfidf_vectors is None:
            self.calculate_tfidf()

        # Prepare tokens and vectors
        query_tokens = self.calculator.preprocess_text(query)
        doc_tokens_list = [
            self.calculator.preprocess_text(ans) for ans in answers]
        query_vector = self.calculator.tfidf_vectors[0]
        doc_vectors = self.calculator.tfidf_vectors[1:]

        # Initialize RelevanceScorer
        self.relevance_scorer = RelevanceScorer(self.calculator)

        # Compute BM25
        self.relevance_scorer.compute_bm25(query_tokens, doc_tokens_list)

        # Compute Cosine Similarity
        self.relevance_scorer.compute_cosine_similarity(
            query_vector, doc_vectors)

        # Compute Answer Length Scores
        self.relevance_scorer.compute_answer_length_scores(doc_tokens_list)

        # Compute Final Relevance Scores
        self.relevance_scorer.compute_relevance_scores()

        # Update Outputs
        self.partb_human_output.setText(
            self.relevance_scorer.get_step_by_step_output())
        self.partb_latex_output.setText(
            self.relevance_scorer.get_step_by_step_latex())

    def save_partb_latex(self):
        """Save Part B LaTeX output to file."""
        if not hasattr(self, 'relevance_scorer'):
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Part B LaTeX File",
            "",
            "LaTeX Files (*.tex);;All Files (*)"
        )

        if file_name:
            if not file_name.endswith('.tex'):
                file_name += '.tex'

            latex_content = (
                "\\documentclass{article}\n"
                "\\usepackage{amsmath}\n"
                "\\begin{document}\n\n"
                f"{self.relevance_scorer.get_step_by_step_latex()}\n"
                "\\end{document}"
            )

            try:
                with open(file_name, 'w') as f:
                    f.write(latex_content)
            except Exception as e:
                print(f"Error saving file: {e}")

    def copy_latex_to_clipboard(self):
        """Copy LaTeX output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.latex_output.toPlainText())

    def copy_human_to_clipboard(self):
        """Copy human-readable output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.human_output.toPlainText())

    def copy_partb_latex_to_clipboard(self):
        """Copy Part B LaTeX output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.partb_latex_output.toPlainText())

    def copy_partb_human_to_clipboard(self):
        """Copy Part B human-readable output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.partb_human_output.toPlainText())


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    window = TFIDFWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
