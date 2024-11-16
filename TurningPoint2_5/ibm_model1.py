from collections import defaultdict
import math


class IBMModel1:
    def __init__(self, eng_sentence, foreign_sentence):
        # Store sentences and vocabularies
        self.eng_sentence = eng_sentence.lower().split()
        self.foreign_sentence = foreign_sentence.lower().split()
        self.eng_vocab = sorted(set(self.eng_sentence))
        self.foreign_vocab = sorted(set(self.foreign_sentence))

        # History tracking
        self.convergence_history = []
        self.iteration_tables = []

    def train(self, iterations=2):
        # Step 1: Initialize translation probabilities uniformly
        t = {}
        for e in self.eng_vocab:
            t[e] = {}
            for f in self.foreign_vocab:
                t[e][f] = 1.0 / len(self.foreign_vocab)

        # Store initial state
        self.convergence_history.append(dict(((e, f), t[e][f])
                                             for e in self.eng_vocab
                                             for f in self.foreign_vocab))

        # EM iterations
        for _ in range(iterations):
            # E-step: Collect counts
            count = defaultdict(lambda: defaultdict(float))
            total = defaultdict(float)

            # For each pair in the sentence
            for e_i, f_i in zip(self.eng_sentence, self.foreign_sentence):
                # Calculate denominator for this English word
                denom = sum(t[e_i][f_j] for f_j in self.foreign_sentence)

                # Update counts
                if denom > 0:  # Avoid division by zero
                    delta = t[e_i][f_i] / denom
                    count[f_i][e_i] += delta
                    total[f_i] += delta

            # M-step: Update probabilities
            for f in self.foreign_vocab:
                for e in self.eng_vocab:
                    if total[f] > 0:
                        t[e][f] = count[f][e] / total[f]

            # Store state for this iteration
            self.convergence_history.append(dict(((e, f), t[e][f])
                                                 for e in self.eng_vocab
                                                 for f in self.foreign_vocab))

            # Calculate perplexity
            perplexity = self._calculate_perplexity(t)
            self.iteration_tables.append({
                'probabilities': {e: dict(t[e]) for e in t},
                'perplexity': perplexity
            })

        return self.convergence_history, self.iteration_tables

    def _calculate_perplexity(self, t):
        """Calculate model perplexity"""
        log_prob = 0.0
        for e, f in zip(self.eng_sentence, self.foreign_sentence):
            prob = t[e][f]
            if prob > 0:
                log_prob += math.log2(prob)
            else:
                log_prob += math.log2(1e-10)  # Smoothing
        return 2 ** (-log_prob / len(self.eng_sentence))

    def generate_latex_table(self, iteration_data):
        """Generate LaTeX code for probability table with perplexity"""
        latex = [
            "\\begin{table}[H]",
            "\\centering",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l|" + "c" * len(self.foreign_vocab) + "}",
            "\\toprule",
            "& " +
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
            "}",
            "\\caption{Translation probabilities and perplexity}",
            "\\end{table}"
        ])

        return "\n".join(latex)

    def generate_convergence_table(self):
        """Generate LaTeX code for convergence table"""
        latex = [
            "\\begin{table}[H]",
            "\\centering",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{ll|" + "c" * len(self.convergence_history) + "}",
            "\\toprule",
            "e & f & initial & " +
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
            "}",
            "\\caption{Convergence of translation probabilities}",
            "\\end{table}"
        ])

        return "\n".join(latex)
