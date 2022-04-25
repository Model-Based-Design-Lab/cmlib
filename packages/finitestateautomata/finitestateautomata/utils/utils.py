""" miscellaneous utility functions """

from string import digits

def warn(s):
    print("Warning: " + s)

def error(s):
    print("Error: "+ s)
    exit()

def inQuotes(state):
    if "," in state:
        return "\"" + state + "\""
    else:
        return state

def printStates(states):
    #quotedStates = list(map(inQuotes, states))
    print(states)

def printSetOfStates(states):
    print("{"+", ".join(states)+"}")
