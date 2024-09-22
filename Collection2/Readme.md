# Unigram and Bigram Calculator

This is a GUI application built with PyQt6 that allows users to input text and calculate the unsmoothed unigram and bigram probabilities. It uses NLTK for tokenization and computes probabilities based on the input text.

## Features

- Calculate unigram probabilities from the input text.
- Calculate bigram probabilities from the input text.
- Display the results in a user-friendly format.
- Supports command-line input for text analysis.

## Requirements

- Python 3.6 or higher
- PyQt6
- NLTK
- absl-py

## Installation

1. **Clone the repository or download the script**:

```bash
git clone https://github.com/iAmGiG/NLPInto.git
cd NLPInto/Collection2
```

2. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

Running the GUI Application
To run the application, execute the following command:

```python
python unigram_bigram_calculator.py
```

This will open the GUI where you can input text and calculate unigram and bigram probabilities.

## Command-Line Input

You can also provide input text via the command line:

```python
python unigram_bigram_calculator.py --input_text="This is a sample text for analysis."
```

The application will open with the provided text already loaded and the results calculated.

After clicking "Calculate", you should see results similar to:

```yaml
Unigram Probabilities:
<s>: 0.0909
the: 0.1818
quick: 0.0909
brown: 0.0909
fox: 0.0909
jumps: 0.0909
over: 0.0909
lazy: 0.0909
dog: 0.0909
</s>: 0.0909

Bigram Probabilities:
<s> the: 1.0000
the quick: 0.5000
quick brown: 1.0000
brown fox: 1.0000
fox jumps: 1.0000
jumps over: 1.0000
over the: 1.0000
the lazy: 0.5000
lazy dog: 1.0000
dog </s>: 1.0000
```

These probabilities represent the likelihood of each word (unigrams) and each word pair (bigrams) occurring in the given text.

### Notes

The application adds start (<s>) and end (</s>) symbols to handle sentence boundaries properly.
The probabilities are unsmoothed and based solely on the frequency counts in the input text.
