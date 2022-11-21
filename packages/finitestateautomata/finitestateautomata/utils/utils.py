""" miscellaneous utility functions """

from string import digits
from typing import List, Set

def warn(s: str):
    print("Warning: " + s)

def error(s: str):
    print("Error: "+ s)
    exit()

def inQuotes(state: str):
    if "," in state:
        return "\"" + state + "\""
    else:
        return state

def printStates(states: List[str]):
    print(states)

def printSetOfStates(states: Set[str]):
    print("{"+", ".join(states)+"}")
