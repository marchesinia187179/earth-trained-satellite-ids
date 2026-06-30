"""
Utility functions for user input management and validation.
"""


# --- Public Functions ---
def validate_choice(value, valid_choices, name="input"):
    """
    Verifies if a value is among the allowed options. Loops until valid

    :param value: string input from the user
    :param valid_choices: list of valid options
    :param name: name of the input for error messages
    :return: validated value (string)
    """
    while value not in valid_choices:
        print(f"Error: Invalid {name} '{value}'. Choose from: {', '.join(valid_choices)}")
        value = input(f"Please re-enter {name}: ").lower()
        
    return value


if __name__ == "__main__":
    pass
