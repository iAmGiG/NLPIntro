# Assuming nlp_a1_q2a contains the updated TextNormalizer
from nlp_a1_q2a import TextNormalizer


def main():
    documents = [
        "the quick brown fox jumps over the lazy dog.",
        "a quick movement of the enemy will jeopardize five gunboats.",
        "five or six big jet planes zoomed quickly by the new tower."
    ]

    # Initialize the TextNormalizer with the documents
    normalizer = TextNormalizer(documents)

    # Process the documents
    normalizer.process_documents()

    # Output the results for each document
    print("Normalized Text, Tokens, and Vocabulary:")
    for i, doc_tokens in enumerate(normalizer.processed_docs, 1):
        print(f"Document {i}:")
        print(f"Tokens: {doc_tokens}")
        print(f"Vocabulary: {normalizer.vocabularies[i-1]}")
        print()

    # Output the combined stats
    print(normalizer.calculate_combined_stats())


if __name__ == '__main__':
    main()
