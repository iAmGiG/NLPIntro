import math
import numpy as np


class RelevanceScorer:
    """Class for handling relevance score calculations with improved LaTeX output."""

    def __init__(self, tfidf_calculator):
        self.tfidf_calculator = tfidf_calculator
        self.bm25_scores = None
        self.cosine_scores = None
        self.length_scores = None
        self.relevance_scores = None
        self.step_by_step_output = ""
        self.step_by_step_latex = ""

    def compute_bm25(self, query_tokens, doc_tokens_list, k=1.5, b=0.75):
        """Compute BM25 scores with LaTeX formatting."""
        N = len(doc_tokens_list)
        avgdl = sum(len(doc_tokens) for doc_tokens in doc_tokens_list) / N
        bm25_scores = []

        # Initialize both outputs
        self.step_by_step_output = "BM25 Calculations:\n"
        self.step_by_step_latex = "\\begin{itemize}\n"
        self.step_by_step_latex += "\\item {BM25 Calculations:}\n"
        self.step_by_step_latex += "\\begin{enumerate}\n"

        for idx, doc_tokens in enumerate(doc_tokens_list):
            score = 0.0
            dl = len(doc_tokens)

            # Start document section
            self.step_by_step_latex += f"    \\item Answer {idx + 1}:\n"
            self.step_by_step_latex += "    \\begin{itemize}\n"

            # Process each term
            for term in set(query_tokens):
                f = doc_tokens.count(term)
                n = sum(1 for tokens in doc_tokens_list if term in tokens)
                idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
                numerator = f * (k + 1)
                denominator = f + k * (1 - b + b * (dl / avgdl))
                term_score = idf * (numerator / denominator)
                score += term_score

                # Add term details to both outputs
                self.step_by_step_output += f"Term '{term}': f={
                    f}, n={n}, idf={idf:.4f}, score={term_score:.4f}\n"
                self.step_by_step_latex += f"        \\item Term '{term}': f={
                    f}, n={n}, idf={idf:.4f}, score={term_score:.4f}\n"

            bm25_scores.append(score)
            # Add total score to both outputs
            self.step_by_step_output += f"Total BM25 Score: {score:.4f}\n\n"
            self.step_by_step_latex += f"        \\item Total BM25 Score: {
                score:.4f}\n"
            self.step_by_step_latex += "    \\end{itemize}\n"

        # Close BM25 section
        self.step_by_step_latex += "\\end{enumerate}\n"
        self.bm25_scores = bm25_scores

    def compute_cosine_similarity(self, query_vector, doc_vectors):
        """Compute cosine similarity scores with LaTeX formatting."""
        cosine_scores = []

        # Start cosine similarity section in both outputs
        self.step_by_step_output += "\nCosine Similarity Calculations:\n"
        self.step_by_step_latex += "\\item Cosine Similarity Calculations:\n"
        self.step_by_step_latex += "\\begin{itemize}\n"

        for idx, doc_vector in enumerate(doc_vectors):
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            score = (dot_product / (norm_query * norm_doc)
                     ) if norm_query and norm_doc else 0.0
            cosine_scores.append(score)

            # Add cosine similarity score to both outputs
            self.step_by_step_output += f"Answer {idx +
                                                  1}: Cosine Similarity = {score:.4f}\n"
            self.step_by_step_latex += f"    \\item Answer {
                idx + 1}: Cosine Similarity = {score:.4f}\n"

        # Close cosine similarity section
        self.step_by_step_latex += "\\end{itemize}\n"
        self.cosine_scores = cosine_scores

    def compute_answer_length_scores(self, doc_tokens_list):
        """Compute normalized answer length scores with LaTeX formatting."""
        lengths = [len(tokens) for tokens in doc_tokens_list]
        max_length = max(lengths)
        length_scores = [length / max_length for length in lengths]
        self.length_scores = length_scores

        # Start length normalization section in both outputs
        self.step_by_step_output += "\nAnswer Length Normalization:\n"
        self.step_by_step_latex += "\\item Answer Length Normalization:\n"
        self.step_by_step_latex += "\\begin{itemize}\n"

        for idx, length in enumerate(lengths):
            normalized_length = length_scores[idx]
            # Add length information to both outputs
            self.step_by_step_output += f"Answer {idx + 1}: Length = {
                length}, Normalized Length = {normalized_length:.4f}\n"
            self.step_by_step_latex += f"    \\item Answer {idx + 1}: Length = {
                length}, Normalized Length = {normalized_length:.4f}\n"

        # Close length normalization section
        self.step_by_step_latex += "\\end{itemize}\n"

    def compute_relevance_scores(self, w1=0.4, w2=0.5, w3=0.1):
        """Compute final relevance scores with LaTeX formatting."""
        relevance_scores = []

        # Start relevance score section in both outputs
        self.step_by_step_output += "\nRelevance Score Calculations:\n"
        self.step_by_step_latex += "\\item Relevance Score Calculations:\n"
        self.step_by_step_latex += "\\begin{itemize}\n"

        for idx in range(len(self.bm25_scores)):
            score = (w1 * self.bm25_scores[idx] +
                     w2 * self.cosine_scores[idx] +
                     w3 * self.length_scores[idx])
            relevance_scores.append(score)

            # Add relevance score calculation to both outputs
            score_text = f"Answer {
                idx + 1}: R = {w1} * BM25 + {w2} * CosSim + {w3} * Length = {score:.4f}\n"
            self.step_by_step_output += score_text
            self.step_by_step_latex += f"    \\item {score_text}"

        # Close relevance score section and entire itemize environment
        self.step_by_step_latex += "\\end{itemize}\n"
        self.step_by_step_latex += "\\end{itemize}"

        self.relevance_scores = relevance_scores

    def get_step_by_step_output(self):
        """Return the step-by-step output in human-readable format."""
        return self.step_by_step_output

    def get_step_by_step_latex(self):
        """Return the step-by-step output in LaTeX format."""
        return self.step_by_step_latex
