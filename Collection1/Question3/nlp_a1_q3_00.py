"""
CFG/PCFG Parser Module

This module provides functionality for parsing sentences using a context-free grammar (CFG) 
or a probabilistic context-free grammar (PCFG). It defines a grammar with production rules 
and probabilities, and allows parsing of input sentences using either CFG or PCFG.

Key Features:
- Parses sentences based on a set of predefined grammar rules.
- Supports both standard CFG parsing and probabilistic parsing (PCFG).
- Provides a graphical user interface (GUI) for manual input and parsing.

Classes:
- TreeNode: Represents a node in the parse tree.
- Parser: Performs the actual parsing based on the grammar and builds the parse tree.
- ParserWindow: Provides a PyQt6-based GUI for user input and displaying the parse tree.

Typical Usage:
    - Use the GUI to input sentences and parse them using CFG or PCFG.
    - Call the Parser class directly for programmatic access to the parsing functionality.
"""
from typing import List, Tuple, Dict
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt6.QtGui import QFont


class TreeNode:
    """
    Represents a node in the parse tree.

    Attributes:
        label (str): The label for this node, typically a grammar symbol (e.g., 'S', 'NP', 'V').
        probability (float): The probability associated with this node (used in PCFG parsing).
        children (List[TreeNode]): A list of child nodes representing the subtree.
    """

    def __init__(self, label: str, probability: float = 1.0):
        """
        Initializes the TreeNode with a label and probability.

        Args:
            label (str): The label for the node (e.g., a grammar symbol).
            probability (float): The probability of this production, default is 1.0.
        """
        self.label = label
        self.probability = probability
        self.children: List[TreeNode] = []

    def add_child(self, child: 'TreeNode'):
        """
        Adds a child node to the current node.

        Args:
            child (TreeNode): The child node to be added to this node.
        """
        self.children.append(child)

    def __str__(self):
        """
        Returns a string representation of the node's label.

        Returns:
            str: The label of the node.
        """
        return self.label


class Parser:
    """
    Parses sentences using a predefined CFG or PCFG grammar.

    This class defines a grammar with production rules and associated probabilities, 
    and provides methods to parse sentences based on the grammar. It supports both
    standard CFG parsing (ignoring probabilities) and probabilistic parsing (PCFG).

    Attributes:
        grammar (Dict[str, List[Tuple[List[str], float]]]): A dictionary representing the grammar 
            where each key is a non-terminal symbol, and each value is a list of tuples representing 
            production rules and their associated probabilities.
    """

    def __init__(self):
        """
        Initializes the parser with the predefined CFG/PCFG grammar.
        Grammar with production rules and probabilities
        """
        self.grammar: Dict[str, List[Tuple[List[str], float]]] = {
            'S': [(['NP', 'VP'], 1.0)],
            'NP': [(['Det', 'N'], 0.6), (['Det', 'Adj', 'N'], 0.4)],
            'VP': [(['V', 'NP'], 0.7), (['V'], 0.3)],
            'Det': [(['the'], 0.6), (['a'], 0.4)],
            'N': [(['cat'], 0.5), (['dog'], 0.5)],
            'Adj': [(['big'], 0.7), (['small'], 0.3)],
            'V': [(['chased'], 0.6), (['slept'], 0.4)]
        }

    def parse(self, sentence: List[str], use_probabilities: bool = False) -> TreeNode:
        """
        Parses a sentence using the CFG or PCFG grammar.

        Args:
            sentence (List[str]): A list of words representing the sentence to be parsed.
            use_probabilities (bool): If True, use PCFG parsing, otherwise use standard CFG parsing.

        Returns:
            TreeNode: The root of the parsed tree if parsing was successful, otherwise None.
        """
        def parse_recursive(symbol: str, words: List[str]) -> Tuple[TreeNode, List[str], float]:
            """
            Recursively parses the sentence based on the grammar rules.

            Args:
                symbol (str): The current grammar symbol being expanded.
                words (List[str]): The remaining words in the sentence to be parsed.

            Returns:
                Tuple[TreeNode, List[str], float]: A tuple containing the parsed subtree, 
                    the remaining words, and the probability of the parse.
            """
            if symbol in self.grammar:
                for production, prob in self.grammar[symbol]:
                    node = TreeNode(symbol, prob if use_probabilities else 1.0)
                    remaining_words = words.copy()
                    all_matched = True
                    total_prob = prob if use_probabilities else 1.0

                    for child_symbol in production:
                        # Recursively parse the child symbol
                        child_node, remaining_words, child_prob = parse_recursive(
                            child_symbol, remaining_words)

                        if child_node is None:
                            all_matched = False
                            break
                        node.add_child(child_node)
                        total_prob *= child_prob

                    # If all symbols matched, return the node and the remaining words
                    if all_matched:
                        return node, remaining_words, total_prob

            # If symbol is not in grammar, treat it as a terminal symbol
            elif words and words[0] == symbol:
                return TreeNode(symbol, 1.0), words[1:], 1.0

            # If no match is found, return None
            return None, words, 0

        # Start the parsing process from the start symbol 'S'
        root, remaining, _ = parse_recursive('S', sentence)
        return root if not remaining else None

    def print_tree(self, node: TreeNode, indent: str = '') -> str:
        """
        Recursively prints the parse tree in a human-readable format.

        Args:
            node (TreeNode): The root of the subtree to be printed.
            indent (str): The current level of indentation for pretty printing.

        Returns:
            str: The formatted string representing the tree structure.
        """
        result = f"{indent}{node.label}"
        if node.probability != 1.0:
            result += f" [{node.probability:.2f}]"
        result += '\n'
        for child in node.children:
            result += self.print_tree(child, indent + '  ')
        return result


class ParserWindow(QMainWindow):
    """
    A GUI window for inputting sentences and parsing them using CFG or PCFG.

    This class provides a graphical user interface where users can enter a sentence, 
    and choose to parse it using either CFG or PCFG. The resulting parse tree is displayed 
    in the GUI.

    Attributes:
        input_text (QTextEdit): A text field for entering the input sentence.
        cfg_button (QPushButton): A button to trigger CFG parsing.
        pcfg_button (QPushButton): A button to trigger PCFG parsing.
        output_text (QTextEdit): A text area to display the parse tree.
        parser (Parser): An instance of the Parser class used to perform the parsing.
    """

    def __init__(self):
        """
        Initializes the GUI components and sets up event handlers for the buttons.
        """
        super().__init__()
        self.setWindowTitle("CFG/PCFG Parser")
        self.setGeometry(500, 300, 880, 620)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter sentence to parse...")
        layout.addWidget(QLabel("Input Sentence:"))
        layout.addWidget(self.input_text)

        self.cfg_button = QPushButton("Parse with CFG")
        self.cfg_button.clicked.connect(lambda: self.parse_sentence(False))
        layout.addWidget(self.cfg_button)

        self.pcfg_button = QPushButton("Parse with PCFG")
        self.pcfg_button.clicked.connect(lambda: self.parse_sentence(True))
        layout.addWidget(self.pcfg_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier"))
        layout.addWidget(QLabel("Parse Tree:"))
        layout.addWidget(self.output_text)

        self.parser = Parser()

    def parse_sentence(self, use_probabilities: bool):
        """
        Parses the sentence entered by the user using either CFG or PCFG.

        Args:
            use_probabilities (bool): If True, parse using PCFG, otherwise parse using CFG.
        """
        sentence = self.input_text.toPlainText().strip().lower().split()
        parse_tree = self.parser.parse(sentence, use_probabilities)
        if parse_tree:
            tree_str = self.parser.print_tree(parse_tree)
            self.output_text.setPlainText(tree_str)
        else:
            self.output_text.setPlainText(
                "Unable to parse the sentence with the given grammar.")


def main():
    """
    Facilitates execution of the application, the gui window, the running logic and loading up display.
    """
    app = QApplication([])
    window = ParserWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
