from collections import defaultdict
import math


class IBMModel1:
    def __init__(self, eng_sentence, foreign_sentence):
        """Initialize IBM Model 1 with sentence pair."""
        self.eng_words = eng_sentence.lower().split()
        self.foreign_words = foreign_sentence.lower().split()
        self.eng_sentence = self.eng_words
        self.foreign_sentence = self.foreign_words
        self.eng_vocab = sorted(set(self.eng_words))
        self.foreign_vocab = sorted(set(self.foreign_words))
        self.convergence_history = []
        self.iteration_tables = []

        print(f"English sentence: {' '.join(self.eng_words)}")
        print(f"Foreign sentence: {' '.join(self.foreign_words)}")

    def train(self, iterations=2):
        """Train IBM Model 1 using EM algorithm."""
        # Step 1: Initialize translation probabilities t(f|e)
        t = defaultdict(lambda: defaultdict(float))
        for e in self.eng_vocab:
            for f in self.foreign_vocab:
                t[e][f] = 1.0 / len(self.foreign_vocab)

        print("\nInitial probabilities:")
        self._print_table(t)
        self.convergence_history.append(self._get_table_for_history(t))

        # EM iterations
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}")

            # Reset counts for this iteration
            count = defaultdict(lambda: defaultdict(float))  # c(e,f)
            total = defaultdict(float)  # total(e)

            # For each sentence pair in corpus (in this case, just one)
            # E-step: Collect counts
            for i, f_i in enumerate(self.foreign_words):
                print(f"\nProcessing foreign word '{f_i}' at position {i}")

                # Calculate normalization for this foreign word
                z = defaultdict(float)  # Store normalization factors
                for j, e_j in enumerate(self.eng_words):
                    z[i] += t[e_j][f_i]

                print(f"Normalization factor z[{i}] = {z[i]:.4f}")

                # Calculate alignment probabilities and update counts
                for j, e_j in enumerate(self.eng_words):
                    if z[i] > 0:  # Prevent division by zero
                        # Calculate delta (expected count) for this word pair
                        delta = t[e_j][f_i] / z[i]

                        # Update counts and totals
                        count[e_j][f_i] += delta
                        total[e_j] += delta

                        print(f"  {e_j}->{f_i}: "
                              f"t={t[e_j][f_i]:.4f}, "
                              f"delta={delta:.4f}, "
                              f"count={count[e_j][f_i]:.4f}, "
                              f"total={total[e_j]:.4f}")

            # M-step: Update translation probabilities
            new_t = defaultdict(lambda: defaultdict(float))
            for e in self.eng_vocab:
                if total[e] > 0:  # Only update if we have counts
                    for f in self.foreign_vocab:
                        # Maximum likelihood estimate
                        new_t[e][f] = count[e][f] / total[e]

            # Store updated probabilities
            t = new_t
            print("\nUpdated probabilities:")
            self._print_table(t)

            # Store state for this iteration
            self.convergence_history.append(self._get_table_for_history(t))
            perplexity = self._calculate_perplexity(t)
            self.iteration_tables.append({
                'probabilities': self._get_table_for_latex(t),
                'perplexity': perplexity
            })
            print(f"Perplexity: {perplexity:.4f}")

        return self.convergence_history, self.iteration_tables

    def _calculate_perplexity(self, t):
        """Calculate perplexity using current parameters."""
        log_prob = 0.0
        for f_i in self.foreign_words:
            # Calculate probability of this foreign word
            prob = 0.0
            for e_j in self.eng_words:
                prob += t[e_j][f_i]

            if prob > 0:
                log_prob += math.log2(prob / len(self.eng_words))
            else:
                log_prob += math.log2(1e-10)  # Smoothing

        return 2 ** (-log_prob / len(self.foreign_words))

    def _print_table(self, t):
        """Print current probability table."""
        print("\nt(f|e) table [rows=e, cols=f]:")
        print(" " * 8, end="")
        for f in self.foreign_vocab:
            print(f"{f:>8}", end=" ")
        print("\n" + "-" * (8 + 9 * len(self.foreign_vocab)))

        for e in self.eng_vocab:
            print(f"{e:8}", end="")
            for f in self.foreign_vocab:
                print(f"{t[e][f]:8.4f}", end=" ")
            print()

    def _get_table_for_history(self, t):
        """Convert probability table for history tracking."""
        return {(e, f): t[e][f]
                for e in self.eng_vocab
                for f in self.foreign_vocab}

    def _get_table_for_latex(self, t):
        """Convert probability table for LaTeX output."""
        return {(e, f): t[e][f]
                for e in self.eng_vocab
                for f in self.foreign_vocab}

    # LaTeX generation methods remain the same...
    def generate_latex_table(self, iteration_data):
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
            row.extend([f"{probs[(e, f)]:.4f}" for f in self.foreign_vocab])
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
