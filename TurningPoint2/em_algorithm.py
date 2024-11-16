# em_algorithm.py
from collections import defaultdict
import copy
from typing import List, DefaultDict, Tuple, Dict


def initialize_probabilities(
    source_tokens: List[str],
    target_tokens: List[str]
) -> DefaultDict[str, DefaultDict[str, float]]:
    """
    Initializes translation probabilities uniformly.
    """
    if not source_tokens or not target_tokens:
        raise ValueError("Source and target tokens cannot be empty")
    t = defaultdict(lambda: defaultdict(float))
    unique_f = set(target_tokens)
    initial_prob = 1.0 / len(unique_f)
    for e in source_tokens:
        for f in unique_f:
            t[f][e] = initial_prob
    return t


def em_step(
    t: DefaultDict[str, DefaultDict[str, float]],
    source_tokens: List[str],
    target_tokens: List[str]
) -> Tuple[
    DefaultDict[str, DefaultDict[str, float]],
    DefaultDict[str, DefaultDict[str, float]],
    Dict[str, float],
    Dict[str, float],
    DefaultDict[str, DefaultDict[str, float]]
]:
    """
    Performs one iteration of the EM algorithm for IBM Model 1.

    Args:
        t: Current translation probability table.
        source_tokens: List of source language tokens.
        target_tokens: List of target language tokens.

    Returns:
        A tuple containing:
        - Updated translation probability table.
        - Count table (c(e_i, f_j)).
        - Normalization factors (s_total(f_j)) for each target word.
        - Total counts (total(e_i)) for each source word.
        - Delta values (delta(e_i, f_j)) for each word pair.
    """
    count = defaultdict(lambda: defaultdict(float))
    total_e = defaultdict(float)
    delta_values = defaultdict(lambda: defaultdict(float))
    normalization_factors = {}

    # Expectation Step
    for f_j in target_tokens:
        s_total_fj = sum(t[f_j][e_i] for e_i in source_tokens)
        normalization_factors[f_j] = s_total_fj
        for e_i in source_tokens:
            delta = t[f_j][e_i] / s_total_fj
            delta_values[e_i][f_j] = delta
            count[e_i][f_j] += delta
            total_e[e_i] += delta

    # Maximization Step
    for e_i in source_tokens:
        for f_j in target_tokens:
            t[f_j][e_i] = count[e_i][f_j] / total_e[e_i]

    return (
        copy.deepcopy(t),
        copy.deepcopy(count),
        normalization_factors,
        total_e,
        delta_values
    )
