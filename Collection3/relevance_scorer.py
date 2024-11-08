import math
import numpy as np


class RelevanceScorer:
    """Class for handling relevance score calculations (Part B)."""

    def __init__(self, tfidf_calculator):
        self.tfidf_calculator = tfidf_calculator
        self.bm25_scores = None
        self.cosine_scores = None
        self.length_scores = None
        self.relevance_scores = None
        self.step_by_step_output = ""
        self.step_by_step_latex = ""

    def compute_bm25(self, query_tokens, doc_tokens_list, k=1.5, b=0.75):
        """Compute BM25 scores."""
        N = len(doc_tokens_list)
        avgdl = sum(len(doc_tokens) for doc_tokens in doc_tokens_list) / N
        bm25_scores = []
        self.step_by_step_output += "BM25 Calculations:\n"
        self.step_by_step_latex += "BM25 Calculations:\n\n"

        for idx, doc_tokens in enumerate(doc_tokens_list):
            score = 0.0
            dl = len(doc_tokens)
            doc_output = f"Answer {idx + 1}:\n"
            doc_latex = f"Answer {idx + 1}:\n\n"
            for term in set(query_tokens):
                f = doc_tokens.count(term)
                n = sum(1 for tokens in doc_tokens_list if term in tokens)
                idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
                numerator = f * (k + 1)
                denominator = f + k * (1 - b + b * (dl / avgdl))
                term_score = idf * (numerator / denominator)
                score += term_score

                # Record step-by-step
                doc_output += f"Term '{term}': f={f}, n={
                    n}, idf={idf:.4f}, score={term_score:.4f}\n"
                doc_latex += f"Term '{term}': f={f}, n={n}, idf={
                    idf:.4f}, score={term_score:.4f}\\\\\n"

            bm25_scores.append(score)
            self.step_by_step_output += doc_output + \
                f"Total BM25 Score: {score:.4f}\n\n"
            self.step_by_step_latex += doc_latex + \
                f"Total BM25 Score: {score:.4f}\\\\\n\n"

        self.bm25_scores = bm25_scores

    def compute_cosine_similarity(self, query_vector, doc_vectors):
        """Compute cosine similarity scores."""
        cosine_scores = []
        self.step_by_step_output += "Cosine Similarity Calculations:\n"
        self.step_by_step_latex += "Cosine Similarity Calculations:\n\n"

        for idx, doc_vector in enumerate(doc_vectors):
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            score = (dot_product / (norm_query * norm_doc)
                     ) if norm_query and norm_doc else 0.0
            cosine_scores.append(score)

            # Record step-by-step
            self.step_by_step_output += f"Answer {idx +
                                                  1}: Cosine Similarity = {score:.4f}\n"
            self.step_by_step_latex += f"Answer {idx +
                                                 1}: Cosine Similarity = {score:.4f}\\\\\n"

        self.cosine_scores = cosine_scores

    def compute_answer_length_scores(self, doc_tokens_list):
        """Compute normalized answer length scores."""
        lengths = [len(tokens) for tokens in doc_tokens_list]
        max_length = max(lengths)
        length_scores = [length / max_length for length in lengths]
        self.length_scores = length_scores

        self.step_by_step_output += "Answer Length Normalization:\n"
        self.step_by_step_latex += "Answer Length Normalization:\n\n"
        for idx, length in enumerate(lengths):
            normalized_length = length_scores[idx]
            self.step_by_step_output += f"Answer {idx + 1}: Length = {
                length}, Normalized Length = {normalized_length:.4f}\n"
            self.step_by_step_latex += f"Answer {idx + 1}: Length = {
                length}, Normalized Length = {normalized_length:.4f}\\\\\n"

    def compute_relevance_scores(self, w1=0.4, w2=0.5, w3=0.1):
        """Compute final relevance scores."""
        relevance_scores = []
        self.step_by_step_output += "Relevance Score Calculations:\n"
        self.step_by_step_latex += "Relevance Score Calculations:\n\n"

        for idx in range(len(self.bm25_scores)):
            score = (w1 * self.bm25_scores[idx] +
                     w2 * self.cosine_scores[idx] +
                     w3 * self.length_scores[idx])
            relevance_scores.append(score)

            # Record step-by-step
            self.step_by_step_output += (f"Answer {idx + 1}: R = {w1} * BM25 + {w2} * CosSim + {w3} * Length = "
                                         f"{score:.4f}\n")
            self.step_by_step_latex += (f"Answer {idx + 1}: R = {w1} * BM25 + {w2} * CosSim + {w3} * Length = "
                                        f"{score:.4f}\\\\\n")

        self.relevance_scores = relevance_scores

    def get_step_by_step_output(self):
        """Return the step-by-step output in human-readable format."""
        return self.step_by_step_output

    def get_step_by_step_latex(self):
        """Return the step-by-step output in LaTeX format."""
        return self.step_by_step_latex
