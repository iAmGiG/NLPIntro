"""
Helps with formatting the tfidf methods.
"""

class TFIDFFormatter:
    @staticmethod
    def generate_latex(vocabulary, tfidf_vectors, text_labels):
        """Generate LaTeX output with improved formatting for large matrices."""
        if not tfidf_vectors or not vocabulary:
            return ""

        # Calculate maximum word length for spacing
        max_word_len = max(len(word) for word in vocabulary)

        # Split vocabulary into manageable chunks
        chunk_size = 8
        vocab_chunks = [list(vocabulary)[i:i + chunk_size]
                        for i in range(0, len(vocabulary), chunk_size)]

        latex_chunks = []
        for _, vocab_chunk in enumerate(vocab_chunks):
            # Generate column headers for the current chunk
            header = " & ".join(word.ljust(max_word_len)
                                for word in vocab_chunk)

            # Generate matrix rows for the current chunk
            rows = []
            for label_idx, label in enumerate(text_labels):
                values = []
                for word in vocab_chunk:
                    word_idx = list(vocabulary).index(word)
                    value = tfidf_vectors[label_idx][word_idx]
                    values.append(f"{value:.3f}")
                rows.append(f"{label} & {' & '.join(values)} \\\\")

            # Combine into a LaTeX matrix
            matrix = (
                "\\begin{array}{c|" + "c" * len(vocab_chunk) + "}\n"
                f" & {header} \\\\ \\hline\n"
                f"{chr(10).join(rows)}\n"
                "\\end{array}"
            )

            latex_chunks.append(matrix)

        # Combine all chunks with spacing between matrices
        final_latex = "\n\\vspace{1em}\n".join(latex_chunks)
        return final_latex

    @staticmethod
    def generate_human_readable(vocabulary, tfidf_vectors, text_labels):
        """Generate human-readable output for the TF-IDF results."""
        if not tfidf_vectors or not vocabulary:
            return ""

        output = []
        vocab_list = list(vocabulary)

        # Create header row
        header = "Word".ljust(20) + "".join(label.ljust(15)
                                            for label in text_labels)
        output.append(header)
        output.append("-" * len(header))

        # Add rows for each word
        for word_idx, word in enumerate(vocab_list):
            row = [word.ljust(20)]
            for vector_idx in range(len(text_labels)):
                value = tfidf_vectors[vector_idx][word_idx]
                row.append(f"{value:.3f}".ljust(15))
            output.append("".join(row))

        return "\n".join(output)
