# data_processing.py
import re


def preprocess_sentence(sentence):
    """
    Tokenizes the sentence into words.
    """
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    return tokens
