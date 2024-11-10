from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTextEdit, QPushButton, QLabel, QFileDialog, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt
from tfidf_calculator import TFIDFCalculator
from tfidf_formatter import TFIDFFormatter
from relevance_scorer import RelevanceScorer

class TFIDFWindow(QMainWindow):
    """Main window for the TF-IDF and Relevance Score Calculator application"""

    def __init__(self):
        super().__init__()
        self.calculator = TFIDFCalculator()
        self.relevance_scorer = None
        self.initUI()

    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('TF-IDF and Relevance Score Calculator')
        self.setGeometry(100, 100, 1000, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create splitters for resizable sections
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Inputs Section
        self.inputs_widget = self.init_inputs_section()
        self.main_splitter.addWidget(self.inputs_widget)

        # Outputs Section
        self.outputs_widget = self.init_outputs_section()
        self.main_splitter.addWidget(self.outputs_widget)

        # Set initial splitter sizes
        self.main_splitter.setSizes([200, 600])

        # Add the splitter to the main layout
        main_layout.addWidget(self.main_splitter)

        # Set default example texts
        self.query_input.setText("What causes global warming?")
        self.answer_inputs[0].setText(
            "Global warming is caused by an increase in greenhouse gases.")
        self.answer_inputs[1].setText(
            "Deforestation contributes to global warming.")
        self.answer_inputs[2].setText(
            "Electric vehicles help reduce carbon emissions.")

    def init_inputs_section(self):
        """Initialize the inputs section"""
        inputs_group = QGroupBox("Inputs")
        inputs_layout = QFormLayout(inputs_group)

        # Query input
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter query here...")
        self.query_input.setFixedHeight(50)
        inputs_layout.addRow("Query:", self.query_input)

        # Answer inputs
        self.answer_inputs = []
        for i in range(3):
            answer_input = QTextEdit()
            answer_input.setPlaceholderText(f"Enter answer {i + 1} here...")
            answer_input.setFixedHeight(50)
            self.answer_inputs.append(answer_input)
            inputs_layout.addRow(f"Answer {i + 1}:", answer_input)

        # Action Buttons
        buttons_layout = QHBoxLayout()
        self.calculate_tfidf_btn = QPushButton("Calculate TF-IDF")
        self.calculate_tfidf_btn.clicked.connect(self.calculate_tfidf)
        buttons_layout.addWidget(self.calculate_tfidf_btn)

        self.calculate_relevance_btn = QPushButton(
            "Calculate Relevance Scores")
        self.calculate_relevance_btn.clicked.connect(
            self.calculate_relevance_scores)
        self.calculate_relevance_btn.setEnabled(False)
        buttons_layout.addWidget(self.calculate_relevance_btn)

        self.calculate_all_btn = QPushButton("Calculate All")
        self.calculate_all_btn.clicked.connect(self.calculate_all)
        buttons_layout.addWidget(self.calculate_all_btn)

        inputs_layout.addRow(buttons_layout)

        return inputs_group

    def init_outputs_section(self):
        """Initialize the outputs section"""
        outputs_splitter = QSplitter(Qt.Orientation.Horizontal)

        # TF-IDF Outputs
        tfidf_outputs_group = QGroupBox("TF-IDF Outputs")
        tfidf_layout = QVBoxLayout(tfidf_outputs_group)

        # Human-readable TF-IDF Output
        self.human_tfidf_output = QTextEdit()
        self.human_tfidf_output.setReadOnly(True)
        tfidf_layout.addWidget(QLabel("Human Readable Output:"))
        tfidf_layout.addWidget(self.human_tfidf_output)

        copy_human_btn = QPushButton("Copy TF-IDF Human Readable to Clipboard")
        copy_human_btn.clicked.connect(self.copy_human_to_clipboard)
        tfidf_layout.addWidget(copy_human_btn)

        # LaTeX TF-IDF Output
        self.latex_tfidf_output = QTextEdit()
        self.latex_tfidf_output.setReadOnly(True)
        tfidf_layout.addWidget(QLabel("LaTeX Output:"))
        tfidf_layout.addWidget(self.latex_tfidf_output)

        copy_latex_btn = QPushButton("Copy TF-IDF LaTeX to Clipboard")
        copy_latex_btn.clicked.connect(self.copy_latex_to_clipboard)
        tfidf_layout.addWidget(copy_latex_btn)

        save_latex_btn = QPushButton("Save TF-IDF LaTeX")
        save_latex_btn.clicked.connect(self.save_latex)
        tfidf_layout.addWidget(save_latex_btn)

        outputs_splitter.addWidget(tfidf_outputs_group)

        # Relevance Score Outputs
        relevance_outputs_group = QGroupBox("Relevance Score Outputs")
        relevance_layout = QVBoxLayout(relevance_outputs_group)

        # Human-readable Relevance Scores Output
        self.human_relevance_output = QTextEdit()
        self.human_relevance_output.setReadOnly(True)
        relevance_layout.addWidget(QLabel("Human Readable Output:"))
        relevance_layout.addWidget(self.human_relevance_output)

        copy_relevance_human_btn = QPushButton(
            "Copy Relevance Scores Human Readable to Clipboard")
        copy_relevance_human_btn.clicked.connect(
            self.copy_relevance_human_to_clipboard)
        relevance_layout.addWidget(copy_relevance_human_btn)

        # LaTeX Relevance Scores Output
        self.latex_relevance_output = QTextEdit()
        self.latex_relevance_output.setReadOnly(True)
        relevance_layout.addWidget(QLabel("LaTeX Output:"))
        relevance_layout.addWidget(self.latex_relevance_output)

        copy_relevance_latex_btn = QPushButton(
            "Copy Relevance Scores LaTeX to Clipboard")
        copy_relevance_latex_btn.clicked.connect(
            self.copy_relevance_latex_to_clipboard)
        relevance_layout.addWidget(copy_relevance_latex_btn)

        save_relevance_latex_btn = QPushButton("Save Relevance Scores LaTeX")
        save_relevance_latex_btn.clicked.connect(self.save_partb_latex)
        relevance_layout.addWidget(save_relevance_latex_btn)

        outputs_splitter.addWidget(relevance_outputs_group)

        # Set initial splitter sizes
        outputs_splitter.setSizes([500, 500])

        return outputs_splitter

    def calculate_tfidf(self):
        """Calculate TF-IDF and update outputs."""
        query = self.query_input.toPlainText()
        answers = [input_field.toPlainText()
                   for input_field in self.answer_inputs]

        texts = [query] + answers
        self.calculator.text_labels = ["Q", "A1", "A2", "A3"]

        # Perform the TF-IDF calculation
        self.calculator.compute_tfidf(texts)

        # Get the vocabulary and TF-IDF vectors
        vocabulary = self.calculator.vocabulary
        tfidf_vectors = self.calculator.tfidf_vectors

        # Update outputs
        self.human_tfidf_output.setText(
            TFIDFFormatter.generate_human_readable(
                vocabulary=vocabulary,
                tfidf_vectors=tfidf_vectors,
                text_labels=self.calculator.text_labels
            )
        )
        self.latex_tfidf_output.setText(
            TFIDFFormatter.generate_latex(
                vocabulary=vocabulary,
                tfidf_vectors=tfidf_vectors,
                text_labels=self.calculator.text_labels
            )
        )

        # Enable Relevance Scores button
        self.calculate_relevance_btn.setEnabled(True)

    def calculate_relevance_scores(self):
        """Calculate relevance scores and update outputs."""
        query = self.query_input.toPlainText()
        answers = [input_field.toPlainText()
                   for input_field in self.answer_inputs]

        # Ensure TF-IDF calculations are done
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

        # Update outputs
        self.human_relevance_output.setText(
            self.relevance_scorer.get_step_by_step_output())
        self.latex_relevance_output.setText(
            self.relevance_scorer.get_step_by_step_latex())

    def calculate_all(self):
        """Perform both TF-IDF and relevance calculations."""
        self.calculate_tfidf()
        self.calculate_relevance_scores()

    def save_latex(self):
        """Save TF-IDF LaTeX output to file."""
        if self.calculator.tfidf_vectors is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save TF-IDF LaTeX File", "", "LaTeX Files (*.tex);;All Files (*)"
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
                f"{self.latex_tfidf_output.toPlainText()}\n"
                "\\end{document}"
            )

            try:
                with open(file_name, 'w') as f:
                    f.write(latex_content)
                print(f"Successfully saved TF-IDF LaTeX file to {file_name}")
            except Exception as e:
                print(f"Error saving file: {e}")

    def save_partb_latex(self):
        """Save Relevance Scores LaTeX output to file."""
        if self.relevance_scorer is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Relevance Scores LaTeX File", "", "LaTeX Files (*.tex);;All Files (*)"
        )

        if file_name:
            if not file_name.endswith('.tex'):
                file_name += '.tex'

            latex_content = (
                "\\documentclass[12pt]{article}\n"
                "\\usepackage{amsmath}\n"
                "\\begin{document}\n\n"
                f"{self.latex_relevance_output.toPlainText()}\n"
                "\\end{document}"
            )

            try:
                with open(file_name, 'w') as f:
                    f.write(latex_content)
                print(f"Successfully saved Relevance Scores LaTeX file to {
                      file_name}")
            except Exception as e:
                print(f"Error saving file: {e}")

    # Copy methods
    def copy_human_to_clipboard(self):
        """Copy TF-IDF human-readable output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.human_tfidf_output.toPlainText())

    def copy_latex_to_clipboard(self):
        """Copy TF-IDF LaTeX output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.latex_tfidf_output.toPlainText())

    def copy_relevance_human_to_clipboard(self):
        """Copy Relevance Scores human-readable output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.human_relevance_output.toPlainText())

    def copy_relevance_latex_to_clipboard(self):
        """Copy Relevance Scores LaTeX output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.latex_relevance_output.toPlainText())

