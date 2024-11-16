# fluency.py
import math
from collections import Counter
from typing import List, Tuple, Dict


def train_language_model(corpus: List[str]) -> Dict[Tuple[str, ...], float]:
    """
    Trains an n-gram language model from the given corpus.

    Args:
        corpus: List of sentences in the target language.

    Returns:
        A dictionary representing the n-gram probabilities.
    """
    n = 3  # Trigram model
    model = Counter()
    total_ngrams = 0
    for sentence in corpus:
        tokens = ['<s>'] + sentence.split() + ['</s>']
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            model[ngram] += 1
            total_ngrams += 1
    # Convert counts to probabilities
    for ngram in model:
        model[ngram] /= total_ngrams
    return model


def calculate_fluency(sentence: str, model: Dict[Tuple[str, ...], float]) -> float:
    """
    Calculates the fluency score of a sentence using the trained language model.

    Args:
        sentence: The sentence to evaluate.
        model: The trained n-gram language model.

    Returns:
        A float representing the log probability of the sentence.
    """
    n = 3
    tokens = ['<s>'] + sentence.split() + ['</s>']
    log_prob = 0.0
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        prob = model.get(ngram, 1e-6)  # Small probability for unseen n-grams
        log_prob += math.log(prob)
    return log_prob
