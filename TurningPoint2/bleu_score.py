from typing import List
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')


def calculate_modified_ngram_precision(candidate: List[str], reference: List[str], n: int) -> float:
    """
    Calculate modified n-gram precision for a given n.
    """
    if len(candidate) < n or len(reference) < n:
        return 0.0

    candidate_ngrams = Counter(zip(*[candidate[i:] for i in range(n)]))
    reference_ngrams = Counter(zip(*[reference[i:] for i in range(n)]))

    total_count = sum(candidate_ngrams.values())
    if total_count == 0:
        return 0.0

    matches = sum(min(count, reference_ngrams[ngram])
                  for ngram, count in candidate_ngrams.items())

    return matches / total_count


def calculate_bleu(reference_sentence: str, candidate_sentence: str) -> float:
    """
    Calculates BLEU score with smoothing for short sentences.

    Args:
        reference_sentence: The reference translation
        candidate_sentence: The candidate translation

    Returns:
        BLEU score between 0 and 1
    """
    # Tokenize sentences
    reference = reference_sentence.split()
    candidate = candidate_sentence.split()

    # Use NLTK's implementation with smoothing
    smoothing = SmoothingFunction()

    # Calculate BLEU score with equal weights for 1-4 grams
    # Uses smoothing method 1 (Adding 1 to both numerator and denominator)
    bleu_score = sentence_bleu(
        [reference],
        candidate,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing.method1
    )

    # For debugging
    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}")

    # Calculate individual n-gram precisions for analysis
    precisions = []
    for n in range(1, 5):
        precision = calculate_modified_ngram_precision(candidate, reference, n)
        precisions.append(precision)
        print(f"{n}-gram precision: {precision:.4f}")

    return bleu_score
