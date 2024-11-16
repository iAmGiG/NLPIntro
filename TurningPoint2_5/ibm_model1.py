from collections import defaultdict
import numpy as np
import math


class IBMModel1:
    def __init__(self, eng_sentence, foreign_sentence):
        # Keep the original sentences as lists of words
        self.eng_sentence = eng_sentence.lower().split()
        self.foreign_sentence = foreign_sentence.lower().split()

        # Get unique words for vocabulary
        self.eng_vocab = sorted(set(self.eng_sentence))
        self.foreign_vocab = sorted(set(self.foreign_sentence))

        self.convergence_history = []
        self.iteration_tables = []

    def train(self, iterations=2):
        # Initialize t(e|f) uniformly
        t = defaultdict(lambda: defaultdict(float))
        for e in self.eng_vocab:
            for f in self.foreign_vocab:
                t[e][f] = 1.0 / len(self.foreign_vocab)

        # Store initial probabilities
        self.convergence_history.append(self._get_probability_table(t))

        # EM iterations
        for iteration in range(iterations):
            # Initialize count tables
            count = defaultdict(lambda: defaultdict(float))
            total = defaultdict(float)

            # E-step: Collect counts
            for e_idx, e in enumerate(self.eng_sentence):
                # Compute normalization for this English word
                denom = sum(t[e][f] for f in self.foreign_sentence)

                # Update counts for all possible foreign words
                for f_idx, f in enumerate(self.foreign_sentence):
                    delta = t[e][f] / denom if denom > 0 else 0
                    count[e][f] += delta
                    total[f] += delta

            # M-step: Estimate probabilities
            for f in self.foreign_vocab:
                for e in self.eng_vocab:
                    if total[f] > 0:
                        t[e][f] = count[e][f] / total[f]
                    else:
                        t[e][f] = 0.0

            # Store probabilities and compute perplexity
            self.convergence_history.append(self._get_probability_table(t))
            perplexity = self._calculate_perplexity(t)

            # Store iteration results
            self.iteration_tables.append({
                'probabilities': dict(t),
                'perplexity': perplexity
            })

        return self.convergence_history, self.iteration_tables

    def _get_probability_table(self, t):
        """Create a dictionary of (e,f) -> probability mappings"""
        return {(e, f): t[e][f] for e in self.eng_vocab for f in self.foreign_vocab}

    def _calculate_perplexity(self, t):
        """Calculate perplexity using the current probability table"""
        log_prob = 0
        sentence_pairs = zip(self.eng_sentence, self.foreign_sentence)

        for e, f in sentence_pairs:
            prob = t[e][f]
            if prob > 0:
                log_prob -= math.log2(prob)
            else:
                # Smoothing for zero probabilities
                log_prob -= math.log2(1e-10)

        return 2 ** (log_prob / len(self.eng_sentence))

    def generate_latex_table(self, iteration_data):
        """Generate LaTeX code for probability table with perplexity"""
        latex = [
            "\\begin{table}[H]",
            "\\centering",
            "\\resizebox{\\textwidth}{!}{%",  # Make table fit page width
            "\\begin{tabular}{l|" + "c" * len(self.foreign_vocab) + "}",
            "\\toprule",
            "& " + \
            " & ".join(
                [f"\\textbf{{{w}}}" for w in self.foreign_vocab]) + "\\\\",
            "\\midrule"
        ]

        probs = iteration_data['probabilities']
        for e in self.eng_vocab:
            row = [e]
            row.extend([f"{probs[e][f]:.4f}" for f in self.foreign_vocab])
            latex.append(" & ".join(row) + "\\\\")

        latex.extend([
            "\\midrule",
            f"\\multicolumn{{{len(self.foreign_vocab) + 1}}}{{r}}{{Perplexity: {
                iteration_data['perplexity']:.2f}}}\\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "}",  # Close resizebox
            "\\caption{Translation probabilities and perplexity}",
            "\\end{table}"
        ])

        return "\n".join(latex)

    def generate_convergence_table(self):
        """Generate LaTeX code for convergence table"""
        latex = [
            "\\begin{table}[H]",
            "\\centering",
            "\\resizebox{\\textwidth}{!}{%",  # Make table fit page width
            "\\begin{tabular}{ll|" + "c" * len(self.convergence_history) + "}",
            "\\toprule",
            "e & f & initial & " + \
            " & ".join(
                [f"it.{i+1}" for i in range(len(self.convergence_history)-1)]) + "\\\\",
            "\\midrule"
        ]

        for e in self.eng_vocab:
            for f in self.foreign_vocab:
                row = [e, f]
                for table in self.convergence_history:
                    row.append(f"{table[(e, f)]:.4f}")
                latex.append(" & ".join(row) + "\\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "}",  # Close resizebox
            "\\caption{Convergence of translation probabilities}",
            "\\end{table}"
        ])

        return "\n".join(latex)
