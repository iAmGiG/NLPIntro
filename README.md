# Regex Utility Module for Text Processing Tasks

This project provides a utility module to solve common text processing tasks using regular expressions (regex) in Python. The module includes solutions for:

1. **Password validation**: Ensuring that a password meets specific complexity requirements.
2. **Subdomain extraction**: Extracting the subdomain from a URL if present.
3. **Phone number extraction**: Extracting US phone numbers from text in various formats.

The project is designed to be simple, extensible, and easily integrated with both command-line or graphical user interfaces.

---

## Features

- **Password validation**: Validate that a password is 8-16 characters long, contains uppercase and lowercase letters, at least one number, and one special character.
- **Subdomain extraction**: Extract the subdomain (if present) from a given URL.
- **Phone number extraction**: Extract US phone numbers in the formats `(123) 456-7890` and `123-456-7890` from any text.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [GUI Option](#gui-option)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/iAmGiG/NLPInto.git
    cd NLPInto
    ```

2. **Set up a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    If you're using the basic text-processing functions, only Python's built-in `re` library is required. However, if you're using the GUI, install the necessary dependencies:

    ```bash
    pip install PyQt6
    ```

### Conda Environment Setup

1. **Create a Conda environment**:

    ```bash
    conda create --name regex-env python=3.9
    ```

2. **Activate the environment**:

    ```bash
    conda activate regex-env
    ```

3. **Install required dependencies**:

    ```bash
    pip install PyQt6 dataclasses absl-py nltk
    ```

4. **Download NLTK data** (if necessary):

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

---

## Usage

You can use the module's functions directly from Python or integrate them into your own projects.

### Password Validation

```python
from regex_util_module import validate_password

password = "Password123!"
is_valid = validate_password(password)
print(f"Is the password valid? {is_valid}")
