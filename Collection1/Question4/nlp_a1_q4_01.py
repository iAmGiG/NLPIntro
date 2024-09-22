from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass
class TreeNode:
    label: str
    children: List['TreeNode']
    span: Tuple[int, int]


class ConstituentEvaluator:
    def __init__(self, gold_sentence: str, predicted_sentence: str):
        self.gold_tree = self.create_gold_tree(gold_sentence)
        self.predicted_tree = self.create_predicted_tree(predicted_sentence)
        self.gold_constituents: Set[Tuple[str, Tuple[int, int]]] = set()
        self.predicted_constituents: Set[Tuple[str, Tuple[int, int]]] = set()

    def create_gold_tree(self, sentence: str) -> TreeNode:
        det1 = TreeNode("Det", [], (0, 1))
        n1 = TreeNode("N", [], (1, 2))
        det_n = TreeNode("Det N", [det1, n1], (0, 2))
        np1 = TreeNode("NP", [det_n], (0, 2))
        v = TreeNode("V", [], (2, 3))
        det2 = TreeNode("Det", [], (3, 4))
        adj = TreeNode("Adj", [], (4, 5))
        n2 = TreeNode("N", [], (5, 6))
        np2 = TreeNode("NP", [det2, adj, n2], (3, 6))
        vp = TreeNode("VP", [v, np2], (2, 6))
        root = TreeNode("S", [np1, vp], (0, 6))
        return root

    def create_predicted_tree(self, sentence: str) -> TreeNode:
        det1 = TreeNode("Det", [], (0, 1))
        n1 = TreeNode("N", [], (1, 2))
        n_parent = TreeNode("N", [det1, n1], (0, 2))
        np1 = TreeNode("NP", [n_parent], (0, 2))
        v = TreeNode("V", [], (2, 3))
        det2 = TreeNode("Det", [], (3, 4))
        adj = TreeNode("Adj", [], (4, 5))
        n2 = TreeNode("N", [], (5, 6))
        np2 = TreeNode("NP", [det2, adj, n2], (3, 6))
        vp = TreeNode("VP", [v, np2], (2, 6))
        root = TreeNode("S", [np1, vp], (0, 6))
        return root

    def extract_constituents(self, node: TreeNode, constituents: Set[Tuple[str, Tuple[int, int]]]):
        if node.children:  # Non-leaf node
            constituents.add((node.label, node.span))
            for child in node.children:
                self.extract_constituents(child, constituents)

    def compute_metrics(self) -> Tuple[int, int, int]:
        self.extract_constituents(self.gold_tree, self.gold_constituents)
        self.extract_constituents(
            self.predicted_tree, self.predicted_constituents)

        tp = len(self.gold_constituents.intersection(
            self.predicted_constituents))
        fp = len(self.predicted_constituents - self.gold_constituents)
        fn = len(self.gold_constituents - self.predicted_constituents)

        return tp, fp, fn

    def calculate_scores(self, tp: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = tp / \
            len(self.predicted_constituents) if len(
                self.predicted_constituents) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1

    def evaluate(self):
        tp, fp, fn = self.compute_metrics()
        accuracy, precision, recall, f1 = self.calculate_scores(tp, fp, fn)

        print(f"Gold Constituents: {self.gold_constituents}")
        print(f"Predicted Constituents: {self.predicted_constituents}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")


# Example usage
evaluator = ConstituentEvaluator(
    "the cat saw a small star", "the cat saw a small star")
evaluator.evaluate()
