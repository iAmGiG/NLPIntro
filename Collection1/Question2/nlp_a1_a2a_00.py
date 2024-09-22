import string
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


class TextNormalizer:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.processed_docs = []
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.combined_tokens = []
        self.combined_vocabulary = set()

    def process_documents(self) -> str:
        result = ""
        for i, doc in enumerate(self.documents, 1):
            result += f"Processing Document {i}\n"
            result += self.process_single_document(doc)
            result += "-" * 50 + "\n"

        result += self.calculate_combined_stats()
        return result

    def process_single_document(self, doc: str) -> str:
        result = f"Original text: {doc}\n\n"

        # Lowercase
        lowercased = doc.lower()
        result += f"Lowercased:\n{lowercased}\n\n"

        # Remove punctuation
        no_punct = lowercased.translate(
            str.maketrans("", "", string.punctuation))
        result += f"Punctuation removed:\n{no_punct}\n\n"

        # Tokenize
        tokens = word_tokenize(no_punct)
        result += f"Tokenized:\n{tokens}\n\n"

        # Remove stop words
        tokens_no_stop = [
            token for token in tokens if token not in self.stop_words]
        result += f"Stop words removed:\n{tokens_no_stop}\n\n"

        # Stem words
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens_no_stop]
        result += f"Stemmed:\n{stemmed_tokens}\n\n"

        # Calculate stats
        vocab = set(stemmed_tokens)
        result += f"Number of tokens: {len(stemmed_tokens)}\n"
        result += f"Number of types: {len(set(stemmed_tokens))}\n"
        result += f"Vocabulary size: {len(vocab)}\n\n"

        self.processed_docs.append(stemmed_tokens)
        self.combined_tokens.extend(stemmed_tokens)
        self.combined_vocabulary.update(vocab)

        return result

    def calculate_combined_stats(self) -> str:
        result = "Combined Statistics\n"
        result += f"Total number of tokens: {len(self.combined_tokens)}\n"
        result += f"Total number of types: {len(set(self.combined_tokens))}\n"
        result += f"Total vocabulary size: {len(self.combined_vocabulary)}\n"
        return result


def main():
    documents = [
        "the quick brown fox jumps over the lazy dog.",
        "a quick movement of the enemy will jeopardize five gunboats.",
        "five or six big jet planes zoomed quickly by the new tower."
    ]
    normalizer = TextNormalizer(documents)
    print(normalizer.process_documents())


if __name__ == '__main__':
    main()
