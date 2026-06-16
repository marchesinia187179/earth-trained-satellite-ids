"""
Utility functions for user input management and validation.
"""
import pathlib
import sys

def validate_path(path_str):
    """Verifies if a path exists and returns the Path object."""
    path = pathlib.Path(path_str)
    if not path.exists():
        print(f"Error: The file path '{path_str}' does not exist.")
        sys.exit(1)
    return path

def get_y_n_choice(prompt):
    """Handles yes/no input with standard validation."""
    choice = input(prompt).lower()
    if choice not in ['y', 'n']:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)
    return choice

def get_y_n_bool(prompt):
    """Returns True if the user chooses 'y', False if 'n'."""
    return get_y_n_choice(prompt) == 'y'

def validate_choice(value, valid_choices, name="input"):
    """Verifies if a value is among the allowed options."""
    if value not in valid_choices:
        print(f"Error: Invalid {name} '{value}'. Choose from: {', '.join(valid_choices)}")
        sys.exit(1)
    return value

def get_split_input(prompt, expected_len):
    """Handles compound inputs (e.g., path and type) and verifies length."""
    user_input = input(prompt).split()
    if len(user_input) != expected_len:
        print(f"Error: Invalid input format. Expected {expected_len} arguments.")
        sys.exit(1)
    return user_input

def get_numeric_input(prompt, type_func=float, min_val=None):
    """Handles numeric input with optional validation on minimum value."""
    try:
        val = type_func(input(prompt))
        if min_val is not None and val <= min_val:
            print(f"Error: The value must be greater than {min_val}.")
            sys.exit(1)
        return val
    except ValueError:
        print(f"Error: Invalid input. Please enter a numeric value.")
        sys.exit(1)