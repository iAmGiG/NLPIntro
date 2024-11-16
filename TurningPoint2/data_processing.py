# data_processing.py
import re
from typing import List


def preprocess_sentence(sentence: str) -> List[str]:
    """
    Tokenizes the sentence into words.

    Args:
        sentence: The sentence to preprocess.

    Returns:
        A list of tokens.
    """
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    return tokens
