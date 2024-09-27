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

# Download necessary NLTK data, ensures that the tokenizastino model,
# which is punkt are avaible for splitting the text into words.
nltk.download('punkt', quiet=True)

# if running in Terminal mode, use `--input_text="[input text]"` flag.
FLAGS = flags.FLAGS
flags.DEFINE_string('input_text', None, 'Input text for analysis')


class UnigramBigramCalculator(QWidget):
    """
    A GUI application to compute unsmoothed unigram and bigram probabilities.
    Default mode is to run in application mode, were a gui is used to input text and recive an result via the output.
    """

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
        self.setGeometry(300, 300, 800, 500)

    def calculate(self):
        """
        Calculate and display unigram and bigram probabilities based on input text.

        Process:
            - retrieve input: `text = self.text_input.toPlainText()` gets the user input from the QT text edit widget.
            - compute Unigrams and Bigrams: calls `self. compute_igrams(text)` to calculate unigrams and bigrams.
            - Format results: calls `self. format_results(unigrams, bigrams)` to format the results into a readable string.
            - Display Results: sets the formatted result string in the `result_display` widget using `self. result_display.setText(result)`.
            """
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

        Purpose: This method computes the unigram and bigram counts from the input text.
        Process:
            - Tokenize text: uses `word_tokenize` from NLTK to split the input text into words. adds start (`<s>`) and ends (`</s>`) 
                tokens to properly handle sentence boundaries.
            - Unigrams: a `defaultdict(int)` to store unigram counts. loops over each token and increments the ecount for that toekn.
            - biigrams: another `defaultdict(int)` stores the bigram counts. looping over consecutive token pairs and increments the count for each pair.
        Return: result list the unigram and bigram dicts, containing the counts.
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

        Process:
            - total unigrams: calcuate the total number of unigrams by summing the values in the `unigrams` dict.
            - unigram probabilites:
                - loops through each token and its count in the `unigram` dict.
                - computes the probability of each unigram: `prob = count / total_unigrams`.
                - appends the token and its probability to the result string.
            - bigram probabilities:
                - loops through each bigram (pain of tokens) and its count in the `bigrams` dict.
                - Computs the probability of each bigram: `prob = count / unigrams[token1]`, 
                    where `token1` is the first token of the bigram.
                - appends the bigram ands it probability to the result string.

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
    The main function to run the GUI application. If the input flag is not none, 
        run the calculator on the input text from the flag.
    A Unigram is made of single words in teh text. 
        the program counts how many times each word appears and calcuates it probabilty 
        by dividign its count by the total number of unigrams.
    A bigram is a pair of consecutive words in the text. the program counts 
        how many times each pair of words appears and calcuates its 
        probability by dividing its count by the count of the first word in the pair.

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
