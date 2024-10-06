import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QMessageBox, QLineEdit, QFileDialog
)
from PyQt6.QtCore import Qt
import numpy as np
import re


class WordEmbeddingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Word Embedding Calculator")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Embeddings Input
        emb_layout = QVBoxLayout()
        emb_label = QLabel("Word Embeddings (format: word: [x1, x2, x3])")
        self.emb_input = QTextEdit()
        self.emb_input.setPlaceholderText("Enter word embeddings here...")
        load_emb_button = QPushButton("Load Embeddings")
        load_emb_button.clicked.connect(self.load_embeddings)
        emb_layout.addWidget(emb_label)
        emb_layout.addWidget(self.emb_input)
        emb_layout.addWidget(load_emb_button)
        layout.addLayout(emb_layout)

        # Negative Samples Input
        neg_layout = QVBoxLayout()
        neg_label = QLabel("Negative Samples (comma-separated words)")
        self.neg_input = QLineEdit()
        self.neg_input.setPlaceholderText("Enter negative samples here...")
        layout.addWidget(neg_label)
        layout.addWidget(self.neg_input)

        # Question A
        q1_layout = QVBoxLayout()
        q1_label = QLabel("Question A: Generate context-target word pairs")
        self.q1_input = QTextEdit()
        self.q1_input.setPlaceholderText("Enter document text here...")
        self.q1_window_size = QLineEdit()
        self.q1_window_size.setPlaceholderText("Enter window size (W)")
        q1_button = QPushButton("Calculate Question A")
        q1_button.clicked.connect(self.calculate_q1)
        q1_layout.addWidget(q1_label)
        q1_layout.addWidget(self.q1_input)
        q1_layout.addWidget(self.q1_window_size)
        q1_layout.addWidget(q1_button)
        layout.addLayout(q1_layout)

        # Question B
        q2_layout = QVBoxLayout()
        q2_label = QLabel("Question B: Word Embedding Update")
        self.q2_target_word = QLineEdit()
        self.q2_target_word.setPlaceholderText(
            "Enter target word (e.g., movies)")
        self.q2_context_word = QLineEdit()
        self.q2_context_word.setPlaceholderText(
            "Enter context word (e.g., watching)")
        q2_button = QPushButton("Calculate Question B")
        q2_button.clicked.connect(self.calculate_q2)
        q2_layout.addWidget(q2_label)
        q2_layout.addWidget(self.q2_target_word)
        q2_layout.addWidget(self.q2_context_word)
        q2_layout.addWidget(q2_button)
        layout.addLayout(q2_layout)

        # Question C
        q3_layout = QVBoxLayout()
        q3_label = QLabel("Question C: Negative Sampling")
        self.q3_target_word = QLineEdit()
        self.q3_target_word.setPlaceholderText(
            "Enter target word (e.g., movies)")
        self.q3_context_word = QLineEdit()
        self.q3_context_word.setPlaceholderText(
            "Enter context word (e.g., watching)")
        q3_button = QPushButton("Calculate Question C")
        q3_button.clicked.connect(self.calculate_q3)
        q3_layout.addWidget(q3_label)
        q3_layout.addWidget(self.q3_target_word)
        q3_layout.addWidget(self.q3_context_word)
        q3_layout.addWidget(q3_button)
        layout.addLayout(q3_layout)

        # Question D
        q4_layout = QVBoxLayout()
        q4_label = QLabel("Question D: Cosine Similarity")
        self.q4_word1 = QLineEdit()
        self.q4_word1.setPlaceholderText("Enter first word (e.g., movies)")
        self.q4_word2 = QLineEdit()
        self.q4_word2.setPlaceholderText("Enter second word (e.g., watching)")
        q4_button = QPushButton("Calculate Question D")
        q4_button.clicked.connect(self.calculate_q4)
        q4_layout.addWidget(q4_label)
        q4_layout.addWidget(self.q4_word1)
        q4_layout.addWidget(self.q4_word2)
        q4_layout.addWidget(q4_button)
        layout.addLayout(q4_layout)

        self.setLayout(layout)

        # Initialize word embeddings
        self.embeddings = {}

    def load_embeddings(self):
        emb_text = self.emb_input.toPlainText()
        pattern = re.compile(r'\"?(\w+)\"?\s*:\s*\[([^\]]+)\]')
        matches = pattern.findall(emb_text)
        if not matches:
            QMessageBox.warning(self, "Input Error",
                                "No valid embeddings found.")
            return
        for word, vector in matches:
            try:
                vector = np.array([float(x.strip())
                                  for x in vector.split(',')])
                self.embeddings[word.lower()] = vector
            except ValueError:
                QMessageBox.warning(self, "Input Error",
                                    f"Invalid vector for word '{word}'.")
                return
        QMessageBox.information(
            self, "Success", "Embeddings loaded successfully.")

    def calculate_q1(self):
        text = self.q1_input.toPlainText().lower()
        W = self.q1_window_size.text()
        if not W.isdigit():
            QMessageBox.warning(self, "Input Error",
                                "Please enter a valid window size.")
            return
        W = int(W)
        words = re.findall(r'\b\w+\b', text)
        pairs = []
        for i, target in enumerate(words):
            start = max(i - W, 0)
            end = min(i + W + 1, len(words))
            context = words[start:i] + words[i+1:end]
            for ctx_word in context:
                pairs.append((target, ctx_word))
        output = "\n".join([f"({t}, {c})" for t, c in pairs])
        QMessageBox.information(self, "Context-Target Pairs", output)

    def calculate_q2(self):
        target_word = self.q2_target_word.text().lower()
        context_word = self.q2_context_word.text().lower()
        learning_rate = 0.01

        if target_word not in self.embeddings or context_word not in self.embeddings:
            QMessageBox.warning(self, "Input Error",
                                "Words not found in embeddings.")
            return

        target_vec = self.embeddings[target_word]
        context_vec = self.embeddings[context_word]

        # Calculate dot product
        dot_product = np.dot(target_vec, context_vec)

        # Sigmoid function
        sigmoid = 1 / (1 + np.exp(-dot_product))

        # Gradient descent update (Skip-gram without negative sampling)
        error = 1 - sigmoid
        gradient = learning_rate * error
        target_vec_updated = target_vec + gradient * context_vec
        context_vec_updated = context_vec + gradient * target_vec

        # Update embeddings
        self.embeddings[target_word] = target_vec_updated
        self.embeddings[context_word] = context_vec_updated

        # Format outputs
        target_vec_formatted = np.round(target_vec_updated, decimals=4)
        context_vec_formatted = np.round(context_vec_updated, decimals=4)

        output = (
            f"Updated embedding for '{target_word}': {target_vec_formatted}\n"
            f"Updated embedding for '{context_word}': {
                context_vec_formatted}\n\n"
            f"Explanation:\n"
            f"Dot product = {dot_product:.4f}\n"
            f"Sigmoid(dot product) = {sigmoid:.4f}\n"
            f"Gradient = learning_rate * (1 - Sigmoid) = {gradient:.4f}"
        )
        QMessageBox.information(self, "Word Embedding Update", output)

    def calculate_q3(self):
        target_word = self.q3_target_word.text().lower()
        context_word = self.q3_context_word.text().lower()
        neg_samples_input = self.neg_input.text().lower()
        negative_samples = [w.strip()
                            for w in neg_samples_input.split(',') if w.strip()]

        if target_word not in self.embeddings or context_word not in self.embeddings:
            QMessageBox.warning(self, "Input Error",
                                "Words not found in embeddings.")
            return

        target_vec = self.embeddings[target_word]

        output_lines = []
        for neg_word in negative_samples:
            neg_vec = self.embeddings.get(neg_word)
            if neg_vec is not None:
                dot_product = np.dot(target_vec, neg_vec)
                output_lines.append(f"Dot product between '{target_word}' and '{
                                    neg_word}': {dot_product:.4f}")
            else:
                output_lines.append(
                    f"Word '{neg_word}' not found in embeddings.")

        # Explanation of negative sampling
        explanation = (
            "\n\nExplanation:\n"
            "Negative sampling helps in efficient training by updating a small subset of weights.\n"
            "It forces the model to distinguish the target-context pair from random noise."
        )

        output = "\n".join(output_lines) + explanation
        QMessageBox.information(self, "Negative Sampling", output)

    def calculate_q4(self):
        word1 = self.q4_word1.text().lower()
        word2 = self.q4_word2.text().lower()

        if word1 not in self.embeddings or word2 not in self.embeddings:
            QMessageBox.warning(self, "Input Error",
                                "Words not found in embeddings.")
            return

        vec1 = self.embeddings[word1]
        vec2 = self.embeddings[word2]

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            cosine_similarity = 0
        else:
            cosine_similarity = dot_product / (norm1 * norm2)

        output = f"Cosine similarity between '{
            word1}' and '{word2}': {cosine_similarity:.4f}"
        QMessageBox.information(self, "Cosine Similarity", output)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WordEmbeddingApp()
    window.show()
    sys.exit(app.exec())
