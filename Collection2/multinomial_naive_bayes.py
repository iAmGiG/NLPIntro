#!/usr/bin/env python3
"""
Multinomial Naive Bayes Classifier GUI Application

This module provides a PyQt6-based GUI application that allows users to input a document and classify it
using a Multinomial Naive Bayes classifier. It shows the probabilities at each step and verifies the answer.

Usage:
    python multinomial_naive_bayes_gui.py
    python multinomial_naive_bayes_gui.py --alpha=1.0
"""

import sys
import math
from collections import defaultdict
from absl import app, flags
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel

FLAGS = flags.FLAGS
flags.DEFINE_float(
    'alpha', 1.0, 'Smoothing parameter for Naive Bayes classifier')


class MultinomialNaiveBayes:
    """A Multinomial Naive Bayes classifier for text documents."""

    def __init__(self):
        """Initialize the classifier with empty counts."""
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()

    def train(self, documents, labels):
        """
        Train the classifier with documents and their corresponding labels.

        Args:
            documents (list of list of str): The training documents.
            labels (list of str): The class labels for each document.
        """
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            for word in doc:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)

    def predict(self, document, alpha=1):
        """
        Predict the class of a document and compute probabilities at each step.

        Args:
            document (list of str): The document to classify.
            alpha (float): Smoothing parameter (default is 1 for Laplace smoothing).

        Returns:
            tuple: A tuple containing the scores and a detailed breakdown of probabilities.
        """
        total_docs = sum(self.class_counts.values())
        scores = {}
        steps = {}

        for label in self.class_counts:
            prior_prob = self.class_counts[label] / total_docs
            prior = math.log(prior_prob)
            likelihood = 0
            likelihood_steps = []
            total_words = sum(self.word_counts[label].values())

            for word in document:
                count = self.word_counts[label].get(word, 0)
                smoothed_prob = (count + alpha) / \
                    (total_words + alpha * len(self.vocabulary))
                likelihood += math.log(smoothed_prob)
                likelihood_steps.append((word, smoothed_prob))

            score = prior + likelihood
            scores[label] = score
            steps[label] = {
                'prior': prior_prob,
                'likelihood_steps': likelihood_steps,
                'score': score,
                'probability': math.exp(score)
            }

        return scores, steps


class NaiveBayesGUI(QWidget):
    """A GUI application for Multinomial Naive Bayes classification."""

    def __init__(self):
        """Initialize the GUI and train the classifier."""
        super().__init__()
        self.initUI()
        self.classifier = MultinomialNaiveBayes()
        self.train_classifier()

    def initUI(self):
        """Set up the GUI layout and widgets."""
        self.setWindowTitle('Multinomial Naive Bayes Classifier')
        self.setGeometry(100, 100, 600, 600)

        layout = QVBoxLayout()

        self.input_label = QLabel(
            'Enter new document (words separated by spaces):')
        layout.addWidget(self.input_label)

        self.text_input = QTextEdit()
        layout.addWidget(self.text_input)

        self.classify_button = QPushButton('Classify')
        self.classify_button.clicked.connect(self.classify)
        layout.addWidget(self.classify_button)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(QLabel('Results:'))
        layout.addWidget(self.result_display)

        self.setLayout(layout)

    def train_classifier(self):
        """Train the Multinomial Naive Bayes classifier with training data."""
        # Training data
        documents = [
            ['fun', 'couple', 'love', 'love'],
            ['fast', 'furious', 'shoot'],
            ['couple', 'fly', 'fast', 'fun', 'fun'],
            ['furious', 'shoot', 'shoot', 'fun'],
            ['fly', 'fast', 'shoot', 'love']
        ]
        labels = ['comedy', 'action', 'comedy', 'action', 'action']

        # Initialize and train the classifier
        self.classifier.train(documents, labels)

    def classify(self):
        """Classify the input document and display probabilities at each step."""
        text = self.text_input.toPlainText()
        document = text.strip().split()

        if not document:
            self.result_display.setText('Please enter a document to classify.')
            return

        alpha = FLAGS.alpha
        scores, steps = self.classifier.predict(document, alpha=alpha)

        result = ''
        for label in scores:
            result += f"Class: {label}\n"
            result += f"Prior probability P({label}) = {
                steps[label]['prior']:.4f}\n"
            result += "Likelihoods:\n"
            for word, prob in steps[label]['likelihood_steps']:
                result += f"P({word}|{label}) = {prob:.4f}\n"
            result += f"Log-score: {steps[label]['score']:.4f}\n"
            result += f"Probability (unnormalized): {
                steps[label]['probability']:.8f}\n"
            result += "\n"

        # Determine the most likely class
        most_likely_class = max(scores, key=scores.get)
        result += f"Most likely class: {most_likely_class}\n"

        self.result_display.setText(result)


def main(argv):
    """
    The main function to run the GUI application.

    Args:
        argv (list): List of command-line arguments.
    """
    qt_app = QApplication(sys.argv)
    gui = NaiveBayesGUI()
    gui.show()
    sys.exit(qt_app.exec())


if __name__ == '__main__':
    app.run(main)
