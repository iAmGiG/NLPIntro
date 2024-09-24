# Multinomial Naive Bayes Classifier GUI Application

This is a GUI application built with PyQt6 that allows users to input a document and classify it using a Multinomial Naive Bayes classifier. It shows the probabilities at each step and verifies the answer.

## Features

- Multinomial Naive Bayes classification of text documents.
- Displays prior probabilities, likelihoods, and log-scores.
- Allows adjustment of the smoothing parameter (alpha) via command-line argument.
- Built with PyQt6 for a user-friendly interface.

## Requirements

- Python 3.x
- PyQt6
- absl-py

## Installation

1. **Clone the repository or download the script**:

```bash
git clone https://github.com/yourusername/multinomial_naive_bayes_gui.git
cd multinomial_naive_bayes_gui
```

## Usage

Running the GUI Application
To run the application, execute the following command:

```bash
python multinomial_naive_bayes_gui.py
```

### Command-Line Arguments

You can adjust the smoothing parameter (alpha) using the --alpha flag:

```bash
python multinomial_naive_bayes_gui.py --alpha=1.0
```

### Example

1. Enter the new document in the text input area:

```shell
fast couple shoot fly
```

2. Click the "Classify" button.

3. View the results, which will include:

- Prior probabilities for each class.
- Likelihoods of each word given the class.
- Log-scores and unnormalized probabilities.
- The most likely class.

### Notes

- The training data is predefined in the code.
- The vocabulary and word counts are built from the training data.
- The probabilities are calculated using Laplace smoothing (add-alpha smoothing).

### Explanation

**Class Priors:**

- Calculated based on the proportion of each class in the training data.
- P(comedy) = Number of comedy documents / Total documents
**Likelihoods:**
- Calculated using Laplace smoothing (add-1 smoothing by default).
- For each word in the document, P(word|class) = (Count of word in class + alpha) / (Total words in class + alpha * Vocabulary size)
- The sum of the log prior and the log likelihoods.
- Used to prevent underflow issues with small probabilities.
**Unnormalized Probabilities:**
- The exponentiated log-scores.
- Not normalized to sum to 1, but can still be used for comparison.
**Most Likely Class:**
- The class with the highest log-score (or highest unnormalized probability).
