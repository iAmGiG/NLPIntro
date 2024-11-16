# em_algorithm.py
from collections import defaultdict
import copy


def initialize_probabilities(source_tokens, target_tokens):
    """
    Initializes translation probabilities uniformly.
    """
    t = defaultdict(lambda: defaultdict(float))
    unique_f = set(target_tokens)
    initial_prob = 1.0 / len(unique_f)
    for e in source_tokens:
        for f in unique_f:
            t[f][e] = initial_prob
    return t


def em_step(t, source_tokens, target_tokens):
    """
    Performs one EM algorithm iteration.
    """
    count = defaultdict(lambda: defaultdict(float))
    total_e = defaultdict(float)
    normalization_factors = {}

    # Expectation Step
    for f in target_tokens:
        normalization_factor = sum(t[f][e] for e in source_tokens)
        normalization_factors[f] = normalization_factor
        for e in source_tokens:
            delta = t[f][e] / normalization_factor
            count[e][f] += delta
            total_e[e] += delta

    # Maximization Step
    for e in total_e:
        for f in count[e]:
            t[f][e] = count[e][f] / total_e[e]

    return copy.deepcopy(t), copy.deepcopy(count), normalization_factors
