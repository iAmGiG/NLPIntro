#!/usr/bin/env python3
"""
Unigram and Bigram Calculator GUI Application

This module provides a PyQt6-based GUI application that allows users to input text and calculate
the unsmoothed unigram and bigram probabilities. It uses NLTK for tokenization and computes
probabilities based on the input text.

Usage:
    python unigram_bigram_calculator.py
    python unigram_bigram_calculator.py --input_text="Sample text here."
"""
import sys
from collections import defaultdict
from absl import app, flags
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

FLAGS = flags.FLAGS
flags.DEFINE_string('input_text', None, 'Input text for analysis')


class UnigramBigramCalculator(QWidget):
    """A GUI application to compute unsmoothed unigram and bigram probabilities."""

    def __init__(self):
        """Initialize the UnigramBigramCalculator GUI."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """Set up the GUI layout and widgets."""
        layout = QVBoxLayout()

        self.text_input = QTextEdit()
        layout.addWidget(QLabel("Enter text:"))
        layout.addWidget(self.text_input)

        calculate_button = QPushButton("Calculate")
        calculate_button.clicked.connect(self.calculate)
        layout.addWidget(calculate_button)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.result_display)

        self.setLayout(layout)
        self.setWindowTitle('Unigram and Bigram Calculator')
        self.setGeometry(300, 300, 400, 400)

    def calculate(self):
        """Calculate and display unigram and bigram probabilities based on input text."""
        text = self.text_input.toPlainText()
        unigrams, bigrams = self.compute_ngrams(text)
        result = self.format_results(unigrams, bigrams)
        self.result_display.setText(result)

    def compute_ngrams(self, text):
        """
        Compute unigram and bigram counts from the input text.

        Args:
            text (str): The input text to analyze.

        Returns:
            tuple: A tuple containing dictionaries of unigram and bigram counts.
        """
        tokens = ['<s>'] + word_tokenize(text.lower()) + ['</s>']

        unigrams = defaultdict(int)
        bigrams = defaultdict(int)

        for token in tokens:
            unigrams[token] += 1

        for i in range(len(tokens) - 1):
            bigrams[(tokens[i], tokens[i+1])] += 1

        return unigrams, bigrams

    def format_results(self, unigrams, bigrams):
        """
        Format the unigram and bigram probabilities into a readable string.

        Args:
            unigrams (dict): Dictionary of unigram counts.
            bigrams (dict): Dictionary of bigram counts.

        Returns:
            str: A formatted string of unigram and bigram probabilities.
        """
        total_unigrams = sum(unigrams.values())

        result = "Unigram Probabilities:\n"
        for token, count in unigrams.items():
            prob = count / total_unigrams
            result += f"{token}: {prob:.4f}\n"

        result += "\nBigram Probabilities:\n"
        for (token1, token2), count in bigrams.items():
            prob = count / unigrams[token1]
            result += f"{token1} {token2}: {prob:.4f}\n"

        return result


def main(argv):
    """
    The main function to run the GUI application.

    Args:
        argv (list): List of command-line arguments.
    """
    app = QApplication(sys.argv)
    calculator = UnigramBigramCalculator()

    if FLAGS.input_text:
        calculator.text_input.setText(FLAGS.input_text)
        calculator.calculate()

    calculator.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    app.run(main)
