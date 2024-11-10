# **TF-IDF and Relevance Score Calculator**

---

## **Task and Purpose Statement**

This project is a Python-based application designed to compute Term Frequency-Inverse Document Frequency (TF-IDF) scores and relevance scores using BM25, cosine similarity, and answer length metrics. The application provides an intuitive graphical user interface (GUI) built with PyQt6, allowing users to enter queries and answers, perform calculations, and view the results in both human-readable and LaTeX formats. The primary goal of this tool is to assist users in text analysis, information retrieval, and natural language processing (NLP) tasks.

---

## **Table of Contents**

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation Instructions](#installation-instructions)
4. [Running the Application](#running-the-application)
5. [Component Overview](#component-overview)
    - [calculator_main.py](#calculator_mainpy)
    - [tfidf_window.py](#tfidf_windowpy)
    - [tfidf_calculator.py](#tfidf_calculatorpy)
    - [tfidf_formatter.py](#tfidf_formatterpy)
    - [relevance_scorer.py](#relevance_scorerpy)
6. [Recommendations and Improvements](#recommendations-and-improvements)

---

## **Features**

- Graphical User Interface (GUI) using PyQt6.
- Calculation of TF-IDF scores for a given query and set of candidate answers.
- Calculation of relevance scores using BM25, cosine similarity, and answer length metrics.
- Resizable and customizable input/output sections for enhanced user experience.
- Export results to LaTeX files and copy outputs to the clipboard for easy sharing.
- Modular design for easy maintainability and scalability.

---

## **Requirements**

- Python 3.8+
- Required Python libraries:

```python
  PyQt6
  numpy
  math
```

```bash
pip install PyQt6 numpy
```

## Running the Application

```bash
python calculator_main.py
```

## Installation Instructions

1. Clone the Repository
2. Install Dependencies
3. Verify the Environment: Ensure that your Python version is 3.8 or higher and all required packages are installed.

## **Component Overview**

### **calculator_main.py**

- **Description**: The main entry point of the application. This script launches the GUI and handles user interactions.
- **Responsibilities**:
  - Initializes the application window and handles input/output sections.
  - Provides buttons for calculating TF-IDF scores, relevance scores, and saving results.
  - Uses the `TFIDFWindow` class to separate logic for the UI.

### **tfidf_window.py**

- **Description**: Contains the `TFIDFWindow` class which defines the main window and user interface.
- **Responsibilities**:
  - Handles the layout, inputs, and outputs for the GUI.
  - Manages interactions between different components (e.g., TF-IDF calculations and relevance scoring).
  - Provides methods for copying results to the clipboard and saving outputs in LaTeX format.

### **tfidf_calculator.py**

- **Description**: Provides methods to compute TF-IDF scores for a set of documents.
- **Responsibilities**:
  - Preprocesses text input by converting it to lowercase and splitting it into words.
  - Computes term frequency (TF) and inverse document frequency (IDF) to generate TF-IDF vectors.
  - Returns TF-IDF vectors for the given query and answers.

### **tfidf_formatter.py**

- **Description**: Formats TF-IDF and relevance scores into human-readable and LaTeX formats.
- **Responsibilities**:
  - Generates formatted LaTeX output for displaying TF-IDF matrices.
  - Creates human-readable output for easier interpretation of results.
  - Provides methods to split large outputs into manageable chunks.

### **relevance_scorer.py**

- **Description**: Computes relevance scores using various metrics like BM25, cosine similarity, and answer length.
- **Responsibilities**:
  - Calculates BM25 scores based on given parameters (k=1.5, b=0.75).
  - Computes cosine similarity between the query and each candidate answer.
  - Computes answer length scores and combines them with other metrics to generate a final relevance score.

---

## **Recommendations and Improvements**

### **System Architect’s Recommendations**

1. **Encapsulation & Code Structure**:
   - Further modularize the code by creating custom widget classes for the inputs and outputs sections.
   - Separate the logic for copying and saving outputs into dedicated utility functions.

2. **Error Handling**:
   - Add validation for empty inputs to prevent runtime errors during calculations.
   - Use `QMessageBox` to display error messages for better user experience.

3. **Scalability**:
   - Implement a configuration file or settings dialog to allow users to adjust parameters (e.g., BM25 weights, chunk size for LaTeX output).
   - Consider adding support for more similarity metrics, such as Jaccard similarity or Word2Vec.

4. **User Interface Enhancements**:
   - Add a progress bar or loading indicator for longer calculations to provide feedback to users.
   - Allow users to dynamically add or remove answer fields instead of having a fixed number.

5. **Testing & Documentation**:
   - Implement unit tests for core functionalities (TF-IDF calculation, BM25, cosine similarity).
   - Add in-code comments and docstrings for better readability and maintainability.
   - Provide a user manual or help section within the application for better onboarding.
