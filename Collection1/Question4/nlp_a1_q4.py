import sys
from typing import List, Tuple, Set
from dataclasses import dataclass
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt6.QtGui import QFont


@dataclass
class TreeNode:
    label: str
    children: List['TreeNode']
    span: Tuple[int, int]


class ConstituentEvaluator:
    def __init__(self, gold_sentence: str, predicted_sentence: str):
        self.gold_tree = self.create_simple_tree(gold_sentence)
        self.predicted_tree = self.create_simple_tree(predicted_sentence)
        self.gold_constituents: Set[Tuple[str, Tuple[int, int]]] = set()
        self.predicted_constituents: Set[Tuple[str, Tuple[int, int]]] = set()

    def create_simple_tree(self, sentence: str) -> TreeNode:
        words = sentence.split()
        leaf_nodes = [TreeNode(word, [], (i, i+1))
                      for i, word in enumerate(words)]
        np_node = TreeNode("NP", leaf_nodes[:2], (0, 2))
        vp_node = TreeNode("VP", leaf_nodes[2:], (2, len(words)))
        root = TreeNode("S", [np_node, vp_node], (0, len(words)))
        return root

    def extract_constituents(self, node: TreeNode, constituents: Set[Tuple[str, Tuple[int, int]]]):
        if not node.children:  # Leaf node
            return

        constituents.add((node.label, node.span))
        for child in node.children:
            self.extract_constituents(child, constituents)

    def compute_metrics(self) -> Tuple[int, int, int, int]:
        self.extract_constituents(self.gold_tree, self.gold_constituents)
        self.extract_constituents(
            self.predicted_tree, self.predicted_constituents)

        tp = len(self.gold_constituents.intersection(
            self.predicted_constituents))
        fp = len(self.predicted_constituents - self.gold_constituents)
        fn = len(self.gold_constituents - self.predicted_constituents)

        sentence_length = max(
            self.gold_tree.span[1], self.predicted_tree.span[1])
        all_possible_spans = sum(
            sentence_length - i for i in range(sentence_length))
        tn = all_possible_spans - (tp + fp + fn)

        return tp, fp, fn, tn

    def calculate_scores(self, tp: int, fp: int, fn: int, tn: int) -> Tuple[float, float, float, float]:
        accuracy = (tp + tn) / (tp + tn + fp +
                                fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1


class ConstituentEvaluatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Constituent Evaluator")
        self.setGeometry(500, 500, 880, 620)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        input_layout = QHBoxLayout()
        layout.addLayout(input_layout)

        self.gold_sentence_input = QTextEdit()
        self.gold_sentence_input.setPlaceholderText(
            "Enter gold standard sentence...")
        input_layout.addWidget(self.gold_sentence_input)

        self.predicted_sentence_input = QTextEdit()
        self.predicted_sentence_input.setPlaceholderText(
            "Enter predicted sentence...")
        input_layout.addWidget(self.predicted_sentence_input)

        self.evaluate_button = QPushButton("Evaluate")
        self.evaluate_button.clicked.connect(self.evaluate_sentences)
        layout.addWidget(self.evaluate_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier"))
        layout.addWidget(QLabel("Evaluation Results:"))
        layout.addWidget(self.output_text)

    def evaluate_sentences(self):
        gold_sentence = self.gold_sentence_input.toPlainText().strip()
        predicted_sentence = self.predicted_sentence_input.toPlainText().strip()

        if not gold_sentence or not predicted_sentence:
            self.output_text.setPlainText(
                "Please enter both gold standard and predicted sentences.")
            return

        evaluator = ConstituentEvaluator(gold_sentence, predicted_sentence)
        tp, fp, fn, tn = evaluator.compute_metrics()
        accuracy, precision, recall, f1 = evaluator.calculate_scores(
            tp, fp, fn, tn)

        result = f"Detailed Matching:\n"
        result += f"Gold Constituents: {evaluator.gold_constituents}\n"
        result += f"Predicted Constituents: {evaluator.predicted_constituents}\n\n"
        result += f"True Positives: {tp}\n"
        result += f"False Positives: {fp}\n"
        result += f"False Negatives: {fn}\n"
        result += f"True Negatives: {tn}\n\n"
        result += f"Accuracy: {accuracy:.4f}\n"
        result += f"Precision: {precision:.4f}\n"
        result += f"Recall: {recall:.4f}\n"
        result += f"F1 Score: {f1:.4f}"

        self.output_text.setPlainText(result)


def main():
    app = QApplication(sys.argv)
    window = ConstituentEvaluatorUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
