# adequacy.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_adequacy(source_sentence: str, translated_sentence: str) -> float:
    """
    Calculates the adequacy score by measuring semantic similarity.

    Args:
        source_sentence: The original sentence in the source language.
        translated_sentence: The translated sentence in the target language.

    Returns:
        A float representing the cosine similarity between the sentences.
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([source_sentence, translated_sentence])
    similarity_matrix = cosine_similarity(tfidf[0:1], tfidf[1:2])
    similarity = similarity_matrix[0][0]
    return similarity
