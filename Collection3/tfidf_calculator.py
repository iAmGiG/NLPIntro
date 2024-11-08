"""
TF-IDF Calculator with GUI Interface

This module implements a Text Frequency-Inverse Document Frequency (TF-IDF) calculator
with a PyQt6-based graphical user interface. It provides functionality to:
    - Calculate TF-IDF vectors for a query and multiple answer texts
    - Display results in both LaTeX and human-readable formats
    - Export calculations to LaTeX files
    - Visualize the relationship between documents based on term importance

The implementation follows the standard TF-IDF formula:
    TF(t,d) = (number of times term t appears in document d) / (total number of terms in document d)
    IDF(t) = log(total number of documents / number of documents containing term t)
    TF-IDF(t,d) = TF(t,d) * IDF(t)

Usage:
    Run the script directly to launch the GUI:
        $ python tfidf_calculator.py
    
    Or import the TFIDFCalculator class to use the calculation functionality:
        from tfidf_calculator import TFIDFCalculator
        calculator = TFIDFCalculator()
        vectors, word_indices = calculator.compute_tfidf(texts)

Dependencies:
    - PyQt6: For the graphical interface
    - numpy: For numerical computations
    - sys: For system-level operations

Authors: iAmGiG
Date: 24H2
Version: 1.0
"""

from collections import Counter
import math
import numpy as np


class TFIDFCalculator:
    """Class for handling TF-IDF calculations"""

    def __init__(self):
        self.tfidf_vectors = None
        self.word_to_idx = None
        self.text_labels = None
        self.word_labels = None

    def preprocess_text(self, text):
        """Convert text to lowercase and split into words."""
        return text.lower().split()

    def compute_tf(self, text):
        """
        Compute term frequencies for a given text document.

        Term Frequency (TF) is calculated as:
            TF(t,d) = (number of occurrences of term t in document d) 
                        / (total number of terms in document d)

        This normalized form ensures that the TF value is not biased by document length.

        Args:
            text (str): The input document text to process.

        Returns:
            dict: A dictionary mapping terms to their frequency scores where:
                - keys are unique terms from the text
                - values are the calculated TF scores (float between 0 and 1)

        Example:
            >>> calculator = TFIDFCalculator()
            >>> text = "the cat and the dog"
            >>> tf = calculator.compute_tf(text)
            >>> print(tf)
            {'the': 0.4, 'cat': 0.2, 'and': 0.2, 'dog': 0.2}
        """
        words = self.preprocess_text(text)
        word_count = Counter(words)
        total_words = sum(word_count.values())
        return {word: count/total_words for word, count in word_count.items()}

    def compute_idf(self, texts):
        """
        Compute inverse document frequencies for terms across all documents.

        IDF is calculated as:
            IDF(t) = log(N / df_t)
        where:
            N = total number of documents
            df_t = number of documents containing term t

        The IDF score decreases the weight of terms that occur frequently across documents
        and increases the weight of terms that occur rarely, capturing term significance.

        Args:
            texts (list of str): List of document texts to process.

        Returns:
            dict: A dictionary mapping terms to their IDF scores where:
                - keys are unique terms from all documents
                - values are the calculated IDF scores

        Note:
            - Uses natural logarithm (ln)
            - Handles edge cases where a term appears in all documents
            - Returns finite scores even for terms appearing in all documents

        Example:
            >>> calculator = TFIDFCalculator()
            >>> texts = ["the cat", "the dog", "some bird"]
            >>> idf = calculator.compute_idf(texts)
            >>> print(idf['the'])  # Term appears in 2 of 3 docs
            0.4054651081081644
        """
        word_doc_count = Counter()
        for text in texts:
            unique_words = set(self.preprocess_text(text))
            word_doc_count.update(unique_words)

        num_docs = len(texts)
        idf = {}
        for word, doc_count in word_doc_count.items():
            idf[word] = math.log(num_docs / doc_count)

        return idf

    def compute_tfidf(self, texts):
        """
        Compute TF-IDF vectors for a collection of texts.

        This method combines Term Frequency (TF) and Inverse Document Frequency (IDF)
        to create a vector representation for each document where:
            TF-IDF(t,d) = TF(t,d) * IDF(t)

        The resulting vectors capture both the importance of terms within individual
        documents (TF) and their discriminative power across the document collection (IDF).

        Args:
            texts (list of str): List of document texts to process.

        Returns:
            tuple: (tfidf_vectors, word_to_idx) where:
                - tfidf_vectors (list of np.array): List of TF-IDF vectors, one per document
                - word_to_idx (dict): Mapping of words to their indices in the vectors

        Features:
            - Creates a consistent vocabulary across all documents
            - Handles missing terms (zero values in vectors)
            - Maintains sparse representation for efficiency
            - Orders terms alphabetically for consistent indexing

        Example:
            >>> calculator = TFIDFCalculator()
            >>> texts = ["the cat", "the dog", "some bird"]
            >>> vectors, word_map = calculator.compute_tfidf(texts)
            >>> print(f"Vocabulary size: {len(word_map)}")
            >>> print(f"Number of vectors: {len(vectors)}")
            >>> print(f"Vector dimensions: {vectors[0].shape}")

        Note:
            The resulting vectors can be used for various NLP tasks such as:
            - Document similarity comparison
            - Information retrieval
            - Document classification
            - Feature extraction for machine learning
        """
        # Get all unique words
        all_words = set()
        for text in texts:
            all_words.update(self.preprocess_text(text))
        self.word_to_idx = {word: i for i,
                            word in enumerate(sorted(all_words))}

        # Compute IDF
        idf = self.compute_idf(texts)

        # Compute TF-IDF for each text
        self.tfidf_vectors = []
        for text in texts:
            tf = self.compute_tf(text)
            tfidf = np.zeros(len(self.word_to_idx))
            for word, tf_value in tf.items():
                if word in self.word_to_idx:
                    idx = self.word_to_idx[word]
                    tfidf[idx] = tf_value * idf[word]
            self.tfidf_vectors.append(tfidf)

        self.word_labels = sorted(self.word_to_idx.keys())
        return self.tfidf_vectors, self.word_to_idx

    def generate_latex(self):
        """
        Generate LaTeX code for displaying the TF-IDF matrix.

        Creates a formatted LaTeX array environment containing the TF-IDF matrix
        with appropriate row and column labels. The output is suitable for
        direct inclusion in a LaTeX document.

        Returns:
            str: LaTeX code representing the TF-IDF matrix where:
                - Rows represent documents (Q, A1, A2, A3)
                - Columns represent terms from the vocabulary
                - Cell values are formatted to 3 decimal places

        Format:
            - Uses array environment for structured layout
            - Includes column separators and horizontal rules
            - Labels rows with document identifiers
            - Labels columns with terms

        Example output format:
            \\begin{array}{c|ccc}
             & term1 & term2 & term3 \\\\ \\hline
            Q & 0.000 & 0.693 & 0.000 \\\\
            A1 & 0.510 & 0.000 & 0.693 \\\\
            ...
            \\end{array}

        Note:
            Requires the TF-IDF calculations to have been performed first.
            Returns an error message if called before calculations.
        """
        if self.tfidf_vectors is None:
            return "No calculations performed yet."

        latex = "\\begin{array}{c|" + "c" * len(self.word_labels) + "}\n"
        latex += " & " + " & ".join(self.word_labels) + " \\\\ \\hline\n"

        for i, row in enumerate(self.tfidf_vectors):
            latex += self.text_labels[i] + " & " + \
                " & ".join(f"{x:.3f}" for x in row) + " \\\\\n"

        latex += "\\end{array}"
        return latex

    def generate_human_readable(self):
        """
        Generate a human-readable string representation of the TF-IDF matrix.

        Creates a formatted text table showing the TF-IDF values in an easily
        readable format with proper alignment and labeling.

        Returns:
            str: A formatted string containing the TF-IDF matrix where:
                - Column headers are term names
                - Row headers are document labels (Q, A1, A2, A3)
                - Values are aligned and formatted to 3 decimal places
                - Includes separator lines for better readability

        Format features:
            - Fixed-width columns for alignment
            - Separator lines between headers and data
            - Right-aligned numerical values
            - Clear column separation using vertical bars

        Example output format:
            TF-IDF Matrix:

            Document |   term1   |   term2   |   term3
            ----------------------------------------
            Q        |    0.000  |    0.693  |    0.000
            A1       |    0.510  |    0.000  |    0.693
            ...

        Note:
            Requires the TF-IDF calculations to have been performed first.
            Returns an error message if called before calculations.
        """
        if self.tfidf_vectors is None:
            return "No calculations performed yet."

        output = "TF-IDF Matrix:\n\n"
        # Header
        output += "Document | " + \
            " | ".join(f"{word:>10}" for word in self.word_labels) + "\n"
        output += "-" * (11 + 13 * len(self.word_labels)) + "\n"

        # Data rows
        for i, row in enumerate(self.tfidf_vectors):
            output += f"{self.text_labels[i]:8} | "
            output += " | ".join(f"{x:10.3f}" for x in row) + "\n"

        return output
