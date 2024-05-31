""" miscellaneous utility functions """

import sys
from typing import List, Set

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
