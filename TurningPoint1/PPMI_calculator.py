import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QTextEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QTableView, QMessageBox,
    QTabWidget, QTextBrowser, QCheckBox
)
from PyQt6.QtCore import QAbstractTableModel, Qt
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import os
from itertools import combinations

# Download NLTK data files if not already present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class PandasModel(QAbstractTableModel):
    """
    A model to interface a pandas DataFrame with QTableView.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the model with a pandas DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to display.
        """
        super().__init__()
        self._data = df

    def rowCount(self, parent=None):
        """
        Return the number of rows in the DataFrame.
        """
        return self._data.shape[0]

    def columnCount(self, parent=None):
        """
        Return the number of columns in the DataFrame.
        """
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """
        Return the data for a given index and role.

        Args:
            index (QModelIndex): The index of the data.
            role (Qt.ItemDataRole): The role for the data.

        Returns:
            Any: The data to display.
        """
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                value = self._data.iloc[index.row(), index.column()]
                if isinstance(value, float):
                    return f"{value:.4f}"  # Format with 4 decimal places
                return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        """
        Return the header data for the table.

        Args:
            section (int): The section (row or column index).
            orientation (Qt.Orientation): Horizontal or Vertical.
            role (Qt.ItemDataRole): The role for the header.

        Returns:
            Any: The header data.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None


class MainWindow(QMainWindow):
    """
    The main window of the PPMI Calculator application.
    """

    def __init__(self):
        """
        Initialize the main window and its components.
        """
        super().__init__()
        self.setWindowTitle("PPMI Calculator")
        
        # Corpus input
        self.corpus_label = QLabel("Corpus (text data):")
        self.corpus_text = QTextEdit()
        self.corpus_text.setPlainText(
            "I love pizza\n"
            "I love pasta\n"
            "I hate broccoli\n"
            "I enjoy pizza and pasta\n"
            "Broccoli is healthy"
        )

        # Vocabulary display
        self.vocab_label = QLabel("Vocabulary:")
        self.vocab_display = QTextEdit()
        self.vocab_display.setReadOnly(True)
        self.vocab_display.setPlainText("")  # Start off blank

        # Checkbox to include/exclude stop words
        self.stopwords_checkbox = QCheckBox("Include stop words")
        # Default is to exclude stop words
        self.stopwords_checkbox.setChecked(False)

        # Calculate button
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_ppmi)

        # Word Pair Analysis Input
        self.word_pairs_label = QLabel(
            "Word Pairs (comma-separated, e.g., 'word1,word2'):")
        self.word_pairs_input = QTextEdit()
        self.word_pairs_input.setPlaceholderText(
            "Enter word pairs, one per line, e.g.\npizza,pasta\nlove,healthy")
        self.word_pair_calculate_button = QPushButton("Analyze Word Pairs")
        self.word_pair_calculate_button.clicked.connect(
            self.analyze_word_pairs)

        # Layout for corpus input and calculate button
        corpus_layout = QVBoxLayout()
        corpus_layout.addWidget(self.corpus_label)
        corpus_layout.addWidget(self.corpus_text)
        corpus_layout.addWidget(self.stopwords_checkbox)
        corpus_layout.addWidget(self.calculate_button)
        corpus_layout.addWidget(self.word_pairs_label)
        corpus_layout.addWidget(self.word_pairs_input)
        corpus_layout.addWidget(self.word_pair_calculate_button)

        # Layout for vocabulary display
        vocab_layout = QVBoxLayout()
        vocab_layout.addWidget(self.vocab_label)
        vocab_layout.addWidget(self.vocab_display)

        # Horizontal layout to place corpus and vocabulary side by side
        input_layout = QHBoxLayout()
        input_layout.addLayout(corpus_layout)
        input_layout.addLayout(vocab_layout)

        # Output area
        self.output_tabs = QTabWidget()

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.output_tabs)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initialize variables
        self.co_occurrence_df = None
        self.probability_matrix_df = None
        self.ppmi_matrix_df = None
        self.count_w_df = None
        self.count_context_df = None
        self.documents = []
        self.vocab = []
        self.occurring_pairs = []  # List to store occurring pairs per sentence

    def calculate_ppmi(self):
        """
        Calculate the PPMI matrix based on the input corpus.
        """
        # Read and process corpus
        corpus_text = self.corpus_text.toPlainText()

        # Tokenize sentences, split by newlines
        sentences = corpus_text.strip().split('\n')

        # Generate vocabulary using NLTK
        include_stopwords = self.stopwords_checkbox.isChecked()
        stop_words = set(stopwords.words('english')
                         ) if not include_stopwords else set()
        all_words = []
        documents = []
        self.occurring_pairs = []  # Reset occurring pairs

        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [word.lower() for word in words if word.isalpha()]
            words = [word for word in words if word not in stop_words]
            all_words.extend(words)
            if words:
                documents.append(words)
                # Generate word pairs for the sentence, excluding self-pairs (w1 != w2)
                pairs = [(w1, w2)
                         for w1, w2 in combinations(words, 2) if w1 != w2]
                formatted_pairs = ', '.join(
                    f"({w1}, {w2})" for w1, w2 in pairs)
                self.occurring_pairs.append(
                    (sentence.strip(), formatted_pairs))
        self.documents = documents  # Store for later use

        # Create vocabulary sorted alphabetically
        vocab = sorted(set(all_words))
        self.vocab = vocab  # Store for later use

        # Update vocabulary display
        self.vocab_display.setPlainText('\n'.join(self.vocab))

        # Create a mapping from word to index
        vocab_to_index = {word: idx for idx, word in enumerate(vocab)}
        vocab_size = len(vocab)

        # Initialize co-occurrence matrix
        co_occurrence = np.zeros((vocab_size, vocab_size), dtype=np.float64)

        # Build co-occurrence matrix
        window_size = 2  # Default size is 2
        for words in documents:
            for idx, word in enumerate(words):
                word_idx = vocab_to_index[word]
                # Define the context window
                start = max(0, idx - window_size)
                end = min(len(words), idx + window_size + 1)
                context_words = words[start:idx] + words[idx+1:end]
                for context_word in context_words:
                    context_idx = vocab_to_index[context_word]
                    co_occurrence[word_idx, context_idx] += 1

        # Proceed with computations
        try:
            # Display Co-occurrence Matrix
            self.co_occurrence_df = pd.DataFrame(
                co_occurrence, index=vocab, columns=vocab)

            # Calculate total sum of co-occurrences
            total_count = np.sum(co_occurrence)

            # Calculate probabilities
            probability_matrix = co_occurrence / total_count
            self.probability_matrix_df = pd.DataFrame(
                probability_matrix, index=vocab, columns=vocab)

            # Calculate count of word (count_w) - Sum over rows
            count_w = np.sum(co_occurrence, axis=1)

            # Calculate count of context (count_context) - Sum over columns
            count_context = np.sum(co_occurrence, axis=0)

            self.count_w_df = pd.DataFrame(
                count_w, index=vocab, columns=["Count_w"])
            self.count_context_df = pd.DataFrame(
                count_context, index=vocab, columns=["Count_context"])

            # Calculate marginal probabilities (row and column sums)
            row_marginal_probs = np.sum(
                probability_matrix, axis=1)  # Sum over rows
            column_marginal_probs = np.sum(
                probability_matrix, axis=0)  # Sum over columns

            # Use epsilon for numerical stability
            epsilon = 1e-8

            # Calculate the PPMI matrix
            ppmi_matrix = np.zeros_like(probability_matrix)

            for i in range(len(vocab)):
                for j in range(len(vocab)):
                    p_ij = probability_matrix[i, j]
                    p_i = row_marginal_probs[i]
                    p_j = column_marginal_probs[j]
                    denominator = p_i * p_j + epsilon  # Ensure numerical stability
                    if p_ij > 0:
                        ppmi = np.log2(p_ij / denominator)  # PMI calculation
                        # Apply the PPMI max(0, PMI) rule
                        ppmi_matrix[i, j] = max(0, ppmi)
                    else:
                        ppmi_matrix[i, j] = 0  # Assign zero if p_ij is zero

            self.ppmi_matrix_df = pd.DataFrame(
                ppmi_matrix, index=vocab, columns=vocab)

            # Display the matrices and analyses
            self.display_matrices()
            self.display_occurring_pairs()
            self.compute_document_vectors()
            # Note: analyze_word_pairs() is now called separately when the user clicks "Analyze Word Pairs"

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred during calculation:\n{e}")

    def display_matrices(self):
        """
        Display the calculated matrices in the GUI.
        """
        # Clear existing tabs
        self.output_tabs.clear()

        # Add human-friendly explanations for each matrix
        matrix_explanations = {
            "Co-occurrence Matrix": "The co-occurrence matrix shows how often words appear together in the given text.",
            "Probability Matrix": "The probability matrix shows the probability of word co-occurrences relative to the total co-occurrences in the text.",
            "PPMI Matrix": "The PPMI matrix shows the positive pointwise mutual information for word pairs, indicating how much more likely two words co-occur than by chance.",
            "Count_w": "Count_w shows how often each word appears as a focus word in the corpus.",
            "Count_context": "Count_context shows how often each word appears as a context word in the corpus."
        }

        matrices = {
            "Co-occurrence Matrix": self.co_occurrence_df,
            "Probability Matrix": self.probability_matrix_df,
            "PPMI Matrix": self.ppmi_matrix_df,
            "Count_w": self.count_w_df,
            "Count_context": self.count_context_df
        }

        for name, df in matrices.items():
            tab = QWidget()
            layout = QVBoxLayout()

            # Explanation label
            explanation = QLabel(matrix_explanations.get(name, ""))
            explanation.setWordWrap(True)
            layout.addWidget(explanation)

            # Table view
            table_view = QTableView()
            model = PandasModel(df)
            table_view.setModel(model)

            # Save button
            save_button = QPushButton(f"Save {name} to CSV")
            save_message = QLabel()
            save_button.clicked.connect(
                lambda checked, df=df, name=name, label=save_message: self.save_matrix(df, name, label))

            layout.addWidget(table_view)
            layout.addWidget(save_button)
            layout.addWidget(save_message)

            tab.setLayout(layout)
            self.output_tabs.addTab(tab, name)

    def display_occurring_pairs(self):
        """
        Display the occurring pairs per sentence in the GUI.
        """
        # Create a text representation of the occurring pairs
        text = ""
        for idx, (sentence, pairs) in enumerate(self.occurring_pairs, start=1):
            text += f"{idx}. \"{sentence}\" - pairs: {pairs}\n\n"

        # Display in a new tab
        tab = QWidget()
        layout = QVBoxLayout()
        text_browser = QTextBrowser()
        text_browser.setText(text.strip())
        layout.addWidget(text_browser)
        tab.setLayout(layout)
        self.output_tabs.addTab(tab, "Occurring Pairs")

    def analyze_word_pairs(self):
        """
        Analyze the association for user-specified word pairs and display the results.
        """
        if self.ppmi_matrix_df is None:
            QMessageBox.warning(
                self, "Warning", "Please calculate the PPMI matrix first by clicking 'Calculate'.")
            return

        # Get word pairs from input
        input_text = self.word_pairs_input.toPlainText()
        lines = input_text.strip().split('\n')
        word_pairs = []
        for line in lines:
            words = line.strip().split(',')
            if len(words) == 2:
                word_pairs.append((words[0].strip(), words[1].strip()))

        if not word_pairs:
            QMessageBox.warning(
                self, "Warning", "Please enter valid word pairs.")
            return

        # Proceed with analysis
        analysis_text = ""
        for word1, word2 in word_pairs:
            if word1 in self.vocab and word2 in self.vocab:
                ppmi_value = self.ppmi_matrix_df.loc[word1, word2]
                analysis_text += f"PPMI({word1}, {word2}) = {ppmi_value:.4f}\n"
                if ppmi_value > 0:
                    analysis_text += f"The positive PPMI value indicates a significant association between '{
                        word1}' and '{word2}'.\n\n"
                else:
                    analysis_text += f"The zero PPMI value indicates little to no association between '{
                        word1}' and '{word2}'.\n\n"
            else:
                analysis_text += f"One or both words '{word1}' or '{
                    word2}' are not in the vocabulary.\n\n"

        # Display in a new tab
        tab = QWidget()
        layout = QVBoxLayout()
        text_browser = QTextBrowser()
        text_browser.setText(analysis_text)
        layout.addWidget(text_browser)
        tab.setLayout(layout)
        self.output_tabs.addTab(tab, "Word Pair Analysis")

    def compute_document_vectors(self):
        """
        Create vector representations for each document and compute cosine similarities.
        """
        # Create a vector for each document
        document_vectors = []
        for doc_words in self.documents:
            # Sum the PPMI vectors of the words in the document
            vector = np.zeros(len(self.vocab))
            for word in doc_words:
                if word in self.vocab:
                    idx = self.vocab.index(word)
                    vector += self.ppmi_matrix_df.iloc[idx].values
            document_vectors.append(vector)

        if len(document_vectors) < 2:
            # Not enough documents to compute similarity
            QMessageBox.warning(
                self, "Warning", "Not enough documents to compute similarity. Please ensure your corpus has at least two documents.")
            return

        # Compute cosine similarity between documents
        similarity_matrix = cosine_similarity(document_vectors)
        similarity_df = pd.DataFrame(similarity_matrix, index=[f"Doc {i+1}" for i in range(len(document_vectors))],
                                     columns=[f"Doc {i+1}" for i in range(len(document_vectors))])

        # Display the similarity matrix
        tab = QWidget()
        layout = QVBoxLayout()

        # Explanation label for cosine similarity
        explanation = QLabel(
            "The cosine similarity scores represent how similar the documents are in terms of word associations.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Table view
        table_view = QTableView()
        model = PandasModel(similarity_df)
        table_view.setModel(model)

        layout.addWidget(table_view)

        # Analysis text
        analysis_text = "Cosine Similarity Scores between Documents:\n\n"
        for i in range(len(document_vectors)):
            for j in range(i+1, len(document_vectors)):
                sim_score = similarity_matrix[i, j]
                analysis_text += f"Similarity between Doc {
                    i+1} and Doc {j+1}: {sim_score:.4f}\n"

        text_browser = QTextBrowser()
        text_browser.setText(analysis_text)
        layout.addWidget(text_browser)

        tab.setLayout(layout)
        self.output_tabs.addTab(tab, "Document Similarity")

    def save_matrix(self, df, name, message_label):
        """
        Save a matrix to a CSV file in the current working directory.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            name (str): The name of the matrix.
            message_label (QLabel): The label to display the save message.
        """
        try:
            file_name = f"{name.lower().replace(' ', '_')}.csv"
            df.to_csv(file_name)
            message = f"Saved to {os.path.abspath(file_name)}"
            message_label.setText(message)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while saving:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
