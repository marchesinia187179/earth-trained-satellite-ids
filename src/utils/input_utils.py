"""
Utility functions for user input management and validation.
"""
import pathlib

def validate_path(path_str):
    """Verifies if a path exists and returns the Path object. Loops until valid."""
    while True:
        path = pathlib.Path(path_str)
        if path.exists():
            return path
        print(f"Error: The file path '{path_str}' does not exist. Please try again.")
        path_str = input("Insert a valid path: ")

def get_y_n_choice(prompt):
    """Handles yes/no input with validation loop."""
    while True:
        choice = input(prompt).lower()
        if choice in ['y', 'n']:
            return choice
        print("Error: Invalid input. Please enter 'y' or 'n'.")

def get_y_n_bool(prompt):
    """Returns True if the user chooses 'y', False if 'n'."""
    return get_y_n_choice(prompt) == 'y'

def validate_choice(value, valid_choices, name="input"):
    """Verifies if a value is among the allowed options. Loops until valid."""
    while value not in valid_choices:
        print(f"Error: Invalid {name} '{value}'. Choose from: {', '.join(valid_choices)}")
        value = input(f"Please re-enter {name}: ").lower()
    return value

def get_split_input(prompt, expected_len):
    """Handles compound inputs and verifies length with a loop."""
    while True:
        user_input = input(prompt).split()
        if len(user_input) == expected_len:
            return user_input
        print(f"Error: Invalid input format. Expected {expected_len} arguments.")

def get_numeric_input(prompt, type_func=float, min_val=None, max_val=None):
    """Handles numeric input with validation loop for min and max bounds."""
    while True:
        try:
            val = type_func(input(prompt))
            # Check range
            if min_val is not None and val < min_val:
                print(f"Error: The value must be greater than or equal to {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Error: The value must be less than or equal to {max_val}.")
                continue
            return val
        except ValueError:
            print("Error: Invalid input. Please enter a numeric value.")