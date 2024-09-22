"""
Regex Utility Module for Various Text Processing Tasks

This module provides functionality to solve several common 
    text processing tasks using regular expressions (regex). 
The tasks include:

1. Password validation: Ensuring that a password meets specific complexity requirements.
2. Subdomain extraction: Extracting the subdomain from a URL if present.
3. Phone number extraction: Extracting phone numbers from a text in various formats.

Each task is implemented using Python's built-in `re` library, 
    and can be easily extended or modified for additional use cases.

Key Functions:
- `validate_password`: Validates that a password meets 
    length, case, digit, and special character requirements.
- `extract_subdomain`: Extracts and returns the subdomain from a given URL.
- `extract_phone`: Extracts US phone numbers in various common formats.

Typical Usage:
    You can use the individual functions for validating passwords, 
        extracting subdomains from URLs, and extracting phone numbers 
    from text. The module can also be integrated with a graphical user interface (GUI) using PyQt6.

Example Usage:

    from regex_util_module import validate_password, extract_subdomain, extract_phone

    # Password validation
    password = "Password123!"
    is_valid = validate_password(password)
    print(f"Is the password valid? {is_valid}")

    # Subdomain extraction
    url = "https://blog.example.com"
    subdomain = extract_subdomain(url)
    print(f"Subdomain: {subdomain}")

    # Phone number extraction
    text = "Call me at (123) 456-7890 or 123-456-7890."
    phone_numbers = extract_phone(text)
    print(f"Extracted phone numbers: {phone_numbers}")
    
Dependencies:
    - `re` library: Python's built-in regex library for performing pattern matching.
    - `PyQt6` (optional): If a GUI is being used for text input and display.
"""
import sys
import re
from absl import app
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QTextEdit


class RegexWindow(QMainWindow):
    """
    supports the creation of the gui for the RegEx tester
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Regex Tester")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Input field
        self.input_field = QLineEdit()
        layout.addWidget(QLabel("Input:"))
        layout.addWidget(self.input_field)

        # Buttons
        button_layout = QHBoxLayout()
        self.password_button = QPushButton("Validate Password")
        self.subdomain_button = QPushButton("Extract Subdomain")
        self.phone_button = QPushButton("Extract Phone Number")
        button_layout.addWidget(self.password_button)
        button_layout.addWidget(self.subdomain_button)
        button_layout.addWidget(self.phone_button)
        layout.addLayout(button_layout)

        # Result display
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(QLabel("Result:"))
        layout.addWidget(self.result_display)

        # Connect buttons to functions
        self.password_button.clicked.connect(self.validate_password)
        self.subdomain_button.clicked.connect(self.extract_subdomain)
        self.phone_button.clicked.connect(self.extract_phone)

    def validate_password(self):
        """
        To validate a password with the requirements of:
        8-16 characters long, 
        contain at least one uppercase letter, 
        one lowercase letter, 
        one number, 
        and one special character.
        we'll need the following:
        ^ - to assert the start of the string.
        (?=.*[A-Z]) - This will ensure we capture at least 1 uppercase character.
        (?=.*[a-z]) - For captureing at least 1 lowercase character.
        (?=.*\d) - to assert at least 1 digit will be captured.
        (?=.*[@$!%*?&]) - ensures at least 1 special chart form the set.
        [A-Za-z\d@$!%*?&]{8,16} - this specified the allowed characters 
            and enforces the length boundaries.
        $ - asserts the end of the string.
        """
        password = self.input_field.text()
        # Password validation regex: 8-16 characters, at least one uppercase letter, one lowercase letter, one number, and one special character
        password_regex = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,16}$'

        if re.match(password_regex, password):
            result = "Password is valid"
        else:
            result = ("Password must be 8-16 characters long,"
                      "\ncontain at least one uppercase letter,"
                      "\none lowercase letter, \none number, \nand one special character.")

        self.result_display.setText(result)

    def extract_subdomain(self):
        """
        - ^(?:https?:\/\/)?: Matches the optional protocol ('http://' or 'https://') at the start of the URL. This is non-capturing, meaning it is matched but not stored in the result.
        - (?:www\.)?: Optionally matches 'www.' at the beginning, but does not capture it as a subdomain.
        - ([^\/\.]+\.)?: Captures the subdomain if it exists. This captures one or more characters (excluding '/' and '.') followed by a period, but it is optional, meaning there may be no subdomain.
        - [^\/\.]+\.[^\/\.]+: Ensures the main domain and top-level domain (e.g., 'example.com') are present.
        """
        url = self.input_field.text()
        # Subdomain extraction regex: extracts the first part of the domain (e.g., blog in blog.example.com)
        subdomain_regex = r'^(?:https?:\/\/)?(?:www\.)?([^\/\.]+\.)?[^\/\.]+\.[^\/\.]+'
        match = re.search(subdomain_regex, url)
        if match:
            potential_subdomain = match.group(1)

            if potential_subdomain:
                # Remove trailing dot
                subdomain = potential_subdomain[:-1]
                # Check if it's not 'www'
                if subdomain.lower() != 'www':
                    result = f"Subdomain: {subdomain}"
                else:
                    result = "No subdomain found (www is not considered a subdomain)"
            else:
                result = "No subdomain found"
        else:
            result = "Invalid URL format"

        self.result_display.setText(result)

    def extract_phone(self):
        """
        \(?\d{3}\)? - Optionally matches an area code in parentheses.
        [-.\s]? - Allows for optional separators such as dashes, dots or spaces.
        \d{3} - matches the next 3 digits.
        [-.\s]? - allows another separator.
        \d{4} - matches the final four digits of the phone number.
        - Supports numbers in the formats:
        - (123) 456-7890
        - 123-456-7890
        - 123.456.7890
        - 123 456 7890
        """
        text = self.input_field.text()
        #  Phone number extraction regex: matches (123) 456-7890 or 123-456-7890 formats
        phone_regex = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        matches = re.findall(phone_regex, text)

        if matches:
            formatted_numbers = []
            for match in matches:
                # Remove all non-numeric characters to normalize the format
                digits = re.sub(r'\D', '', match)

                # Format the phone number as (123) 456-7890
                formatted_number = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                formatted_numbers.append(formatted_number)

            result = "Phone numbers found: " + ", ".join(formatted_numbers)
        else:
            result = "No phone numbers found or invalid format."
        self.result_display.setText(result)


def main(argv):
    app = QApplication(sys.argv)
    window = RegexWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    app.run(main)
