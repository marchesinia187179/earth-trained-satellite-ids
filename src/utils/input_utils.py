import pathlib
import sys

def validate_path(path_str):
    """Verifica se un percorso esiste e restituisce l'oggetto Path."""
    path = pathlib.Path(path_str)
    if not path.exists():
        print(f"Error: The file path '{path_str}' does not exist.")
        sys.exit(1)
    return path

def get_y_n_choice(prompt):
    """Gestisce l'input yes/no con validazione standard."""
    choice = input(prompt).lower()
    if choice not in ['y', 'n']:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)
    return choice

def get_y_n_bool(prompt):
    """Restituisce True se l'utente sceglie 'y', False se 'n'."""
    return get_y_n_choice(prompt) == 'y'

def validate_choice(value, valid_choices, name="input"):
    """Verifica se un valore è tra le opzioni consentite."""
    if value not in valid_choices:
        print(f"Error: Invalid {name} '{value}'. Choose from: {', '.join(valid_choices)}")
        sys.exit(1)
    return value

def get_split_input(prompt, expected_len):
    """Gestisce input composti (es. path e tipo) e ne verifica la lunghezza."""
    user_input = input(prompt).split()
    if len(user_input) != expected_len:
        print(f"Error: Invalid input format. Expected {expected_len} arguments.")
        sys.exit(1)
    return user_input

def get_numeric_input(prompt, type_func=float, min_val=None):
    """Gestisce l'input numerico con validazione opzionale sul valore minimo."""
    try:
        val = type_func(input(prompt))
        if min_val is not None and val <= min_val:
            print(f"Error: The value must be greater than {min_val}.")
            sys.exit(1)
        return val
    except ValueError:
        print(f"Error: Invalid input. Please enter a numeric value.")
        sys.exit(1)