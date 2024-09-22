"""
Text Normalization and GUI Interface Module

This module provides functionality for normalizing text by performing common
text preprocessing steps, including:
- Lowercasing text
- Removing punctuation
- Tokenizing the text into words
- Removing stop words
- Stemming the tokens to their base form
- Calculating basic statistics such as the number of tokens and vocabulary size

Additionally, the module provides a graphical user interface (GUI) for manual 
text input and real-time processing. The user can input a document, view each 
step of the normalization process, and see combined stats across multiple 
documents processed through the interface.

Key Classes:
- TextNormalizer: Handles all the text normalization processes, including single-document 
  and multi-document processing. It encapsulates the entire normalization pipeline.
  
- TextNormalizerWindow: A GUI window that allows users to input text, process it, 
  and display the intermediate steps and final statistics. The class tracks tokens across 
  all processed documents and provides combined stats.

Key Features:
- Manual mode (GUI): Allows for manual input and processing of documents through the 
  graphical interface. Displays intermediate steps of the text normalization process.
  
- Auto mode: Processes multiple predefined documents and outputs combined stats such as 
  total tokens and combined types across all documents.

Typical Usage:
To use the module in manual mode (GUI), run the script with the '--mode=manual' flag.
For auto mode, which processes predefined documents, use the '--mode=auto' flag.

Example:

    python your_script.py --mode=manual   # Launch the GUI
    python your_script.py --mode=auto     # Run auto mode for batch processing

"""
import string
from typing import List
from absl import app, flags
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'mode', 'manual', 'Mode: "manual" for GUI mode or "auto" for processing multiple documents')


class TextNormalizerWindow(QMainWindow):
    """
    A class used to create a GUI for inputting and processing text using the TextNormalizer class.

    This class provides a graphical interface where users can input text, process it, and view the
    intermediate and final results. It supports manual text entry and displays each step of the 
    text normalization process, including lowercasing, removing punctuation, tokenizing, removing stop
    words, stemming, and calculating stats.

    Attributes:
        input_text (QTextEdit): A text field for the user to input text.
        process_button (QPushButton): A button to trigger the text processing.
        output_text (QTextEdit): A text field to display the results.
        combined_tokens (list): A list to store the tokens from all documents processed via the GUI.
    """
    def __init__(self):
        """
        Initializes the TextNormalizerWindow class for creating the GUI.
        
        Attributes:
            input_text (QTextEdit): A text field for the user to input text.
            process_button (QPushButton): A button to trigger the text processing.
            output_text (QTextEdit): A text field to display the results.
            combined_tokens (list): A list to store the tokens from all documents processed via the GUI.
        """
        super().__init__()
        self.setWindowTitle("Text Normalizer")
        self.setGeometry(500, 300, 1050, 620)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter text here...")
        layout.addWidget(QLabel("Input Text:"))
        layout.addWidget(self.input_text)

        self.process_button = QPushButton("Process Text")
        self.process_button.clicked.connect(self.process_text)
        layout.addWidget(self.process_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_text)

    def process_text(self):
        """
        Processes the text entered in the input field using the TextNormalizer class.
        Displays the intermediate steps and final statistics, including total combined types from
        all documents processed in the GUI.
        
        Intermediate results include:
            - Lowercased text
            - Text with punctuation removed
            - Tokenized text
            - Text with stop words removed
            - Stemmed tokens
            - Document statistics (number of tokens and vocabulary size)
        
        Additionally, the total combined types across all processed documents are displayed.
        """
        input_text = self.input_text.toPlainText()
        normalizer = TextNormalizer()

        # Display the original text
        result = f"Original text: {input_text}\n\n"

        # Perform each step and show intermediate results
        lowercased = normalizer.lowercase(input_text)
        result += f"Lowercased: {lowercased}\n\n"

        no_punctuation = normalizer.remove_punctuation(lowercased)
        result += f"Removed punctuation: {no_punctuation}\n\n"

        tokens = normalizer.tokenize(no_punctuation)
        result += f"Tokens: {tokens}\n\n"

        no_stop_words = normalizer.remove_stop_words(tokens)
        result += f"Removed stop words: {no_stop_words}\n\n"

        stemmed_tokens = normalizer.stem_words(no_stop_words)
        result += f"Stemmed words: {stemmed_tokens}\n\n"

        # Calculate stats and show
        stats = normalizer.calculate_stats(stemmed_tokens)
        result += f"\n{stats}"

        self.output_text.setPlainText(result)


class TextNormalizer:
    """
    A class used to normalize text by performing common text processing tasks such as:
    - Lowercasing the text
    - Removing punctuation
    - Tokenizing the text into words
    - Removing stop words
    - Stemming the words to their base form
    
    This class supports both single-document processing and multiple-document processing.

    Attributes:
        stop_words (set): A set of stop words loaded from the NLTK stopwords corpus.
        stemmer (PorterStemmer): An instance of the PorterStemmer for stemming words.
    """
    def __init__(self):
        """
        Initializes the TextNormalizer class.

        Attributes:
            stop_words (set): A set of stop words loaded from the NLTK stopwords corpus.
            stemmer (PorterStemmer): An instance of the PorterStemmer for stemming words.
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def lowercase(self, text: str) -> str:
        """
        Converts the input text to lowercase.
        
        Args:
            text (str): The input text to be processed.
        
        Returns:
            str: The lowercase version of the input text.
        """
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """
        Removes punctuation from the input text.
        
        Args:
            text (str): The input text to be processed.
        
        Returns:
            str: The text with punctuation removed.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into individual words (tokens).
        
        Args:
            text (str): The input text to be tokenized.
        
        Returns:
            list: A list of tokens (words) from the input text.
        """
        return word_tokenize(text)

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Removes stop words from the tokenized input.
        
        Args:
            tokens (list): A list of tokenized words (tokens).
        
        Returns:
            list: A list of tokens with stop words removed.
        """
        return [token for token in tokens if token not in self.stop_words]

    def stem_words(self, tokens: List[str]) -> List[str]:        
        """
        Applies stemming to the input tokens.
        
        Args:
            tokens (list): A list of tokenized words.
        
        Returns:
            list: A list of stemmed tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]

    def calculate_stats(self, tokens: List[str]) -> str:
        """
        Calculates the number of tokens and vocabulary size (unique words) for the given tokens.
        
        Args:
            tokens (list): A list of tokens to calculate stats for.
        
        Returns:
            str: A formatted string showing the number of tokens and vocabulary size.
        """
        num_tokens = len(tokens)
        num_types = len(set(tokens))
        return f"Number of tokens: {num_tokens}\nVocabulary size: {num_types}"

    def process_multiple_documents(self, documents: List[str]):
        """
        Processes multiple documents by applying the text normalization steps to each.
        Returns the combined tokens and vocabulary for all documents.
        
        Args:
            documents (list): A list of documents (strings) to be processed.
        
        Returns:
            tuple: 
                - all_processed_docs (list): A list of lists, where each inner list contains the tokens from a document.
                - combined_tokens (list): A list of all tokens from all documents combined.
                - combined_vocabulary (set): A set of unique tokens (vocabulary) across all documents.
        """
        combined_tokens = []
        combined_vocabulary = set()
        all_processed_docs = []

        for doc in documents:
            tokens = self.tokenize(
                self.remove_punctuation(self.lowercase(doc)))
            tokens = self.remove_stop_words(tokens)
            tokens = self.stem_words(tokens)

            all_processed_docs.append(tokens)
            combined_tokens.extend(tokens)
            combined_vocabulary.update(tokens)

        # Returning three values: processed docs, combined tokens, and combined vocabulary
        return all_processed_docs, combined_tokens, combined_vocabulary


class TextNormalizerController:
    """
    Helps with controlling the auto and manual mode interfaces.
    """
    def __init__(self):
        self.normalizer = TextNormalizer()

    def run_manual_mode(self, argv):
        """
        enables the gui interface and actions executing the application.
        """
        app = QApplication(argv)
        window = TextNormalizerWindow()
        window.show()
        app.exec()

    def run_auto_mode(self):
        """
        if in auto mode, use the documents list to evaluate several strings at a time.
        """
        documents = [
            "the quick brown fox jumps over the lazy dog.",
            "a quick movement of the enemy will jeopardize five gunboats.",
            "five or six big jet planes zoomed quickly by the new tower."
        ]

        # Call process_multiple_documents and unpack the three returned values
        processed_docs, combined_tokens, combined_vocabulary = self.normalizer.process_multiple_documents(
            documents)

        for i, tokens in enumerate(processed_docs, 1):
            print(f"Document {i} Tokens: {tokens}")
            print(f"Document {i} Vocabulary: {set(tokens)}\n")

        # Now print combined stats
        print(f"Total combined tokens: {len(combined_tokens)}")
        print(f"Total combined types: {len(set(combined_tokens))}")
        print(f"Total combined vocabulary size: {len(combined_vocabulary)}")


def main(argv):
    """
    Using manual will introduce a gui envrionment to test a single document.
    Using auto will enable a multi document model, but has to be manual inserted into code for now.
    """
    controller = TextNormalizerController()

    # Check mode from flags
    if FLAGS.mode == 'manual':
        controller.run_manual_mode(argv)
    elif FLAGS.mode == 'auto':
        controller.run_auto_mode()


if __name__ == '__main__':
    app.run(main)
