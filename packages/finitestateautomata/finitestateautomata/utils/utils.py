""" miscellaneous utility functions """

import sys
from typing import List, Set

class FiniteStateAutomataException(Exception):
    """Exceptions related to this package"""

def warn(s: str):
    '''Print warning.'''
    print("Warning: " + s)

def error(s: str):
    '''Print error.'''
    print("Error: "+ s)
    sys.exit()

def in_quotes(state: str):
    '''Return string within double quotes.'''
    if "," in state:
        return "\"" + state + "\""
    return state

def print_states(states: List[str]):
    '''Print list of states.'''
    print(states)

def print_set_of_states(states: Set[str]):
    '''Print a set of states.'''
    print("{"+", ".join(states)+"}")

def string_hash(input_string: str)->int:
    """
    a deterministic string hash, because Python's default is not it seems.
    """
    prime = 31
    hash_value = 0

    for char in input_string:
        # Multiply the current hash value by the prime
        # and add the ASCII value of the character
        hash_value = (hash_value * prime + ord(char)) % (2**64)  # To keep it within 64 bits

    return hash_value