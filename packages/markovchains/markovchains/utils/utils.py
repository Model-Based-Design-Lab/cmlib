""" miscellaneous utility functions """

from string import digits
from fractions import Fraction
import numpy as np

Frac = lambda n : Fraction(n).limit_denominator(max_denominator=1000000) if not np.isnan(n) else n

def warn(s):
    print("Warning: " + s)

def error(s):
    print("Error: "+ s)
    exit()

def onlyDigits(s):
    return ''.join(c for c in s if c in digits)

def onlyNonDigits(s):
    return ''.join(c for c in s if c not in digits)


def getIndex(name):
    dig = onlyDigits(name)
    if dig == '':
        return -1
    else:
        return int(dig)

def isNumerical(setOfNames):
    alphNames = set([onlyNonDigits(s) for s in setOfNames])
    return len(alphNames) <= 1

def stringToFloat(s, default):
    try:
        return float(s)
    except ValueError:
        return default


def sortNames(setOfNames):

    listOfNames = list(setOfNames)
    if isNumerical(setOfNames):
        listOfNames.sort(key=getIndex)
    else:
        listOfNames.sort()
    return listOfNames

def printList(l):
    try:
        string = "["
        for item in l:
            string += "{:.4f}, ".format(item)
        return string[:-2] + "]"
    except:
        return l

def printListFrac(l):
    try:
        string = "[ "
        for item in l:
            string += "{} ".format(Frac(item))
        return string + "]"
    except:
        return l

def printDListFrac(dl):
    try:
        string = "[\n"
        for l in dl:
            string += "[ "
            for item in l:
                string += "{} ".format(Frac(item))
            string = string + "] \n"
        return string[:-2] + "\n]"
    except:
        return dl

def printDList(dl):
    try:
        string = "["
        for l in dl:
            string += "["
            for item in l:
                string += "{:.4f}, ".format(item)
            string = string[:-2] + "], "
        return string[:-2] + "]"
    except:
        return dl

def print4F(nr):
    try:
        return "{:.4f}".format(nr)
    except:
        return "{}".format(nr)

def printSortedSet(s):
    print("{{{}}}".format(", ".join(sortNames(s))))

def printSortedList(s):
    print("{}".format(", ".join(sortNames(s))))

def printVector(s):
    print ("[{}]\n".format(", ".join(s)))


def stopCriteria(c):
    # Format stop criteria: 
    stop = '''
    Steady state behaviour = 
    [
        Confidence level,
        Absolute error,
        Relative error,
        Maximum path length,
        Maximum number of cycles,
        Time (in seconds)
    ]

    Transient behaviour = 
    [
        Confidence level,
        Absolute error,
        Relative error,
        Maximum path length,
        Maximum number of paths,
        Time (in seconds)
    ]
    '''

    if len(c) != 6:
        s = "Wrong number of arguments for -c\n" + stop
        error(s)
    
    # Confidence level
    if c[0] <= 0:
        s = "No confidence level set"
        error(s)
    
    if all(i<=0 for i in c[1:]):
        s = "No stop conditions inside -c argument, simulation never terminating"
        error(s)
    
    return c

def nrOfSteps(ns):
    if ns == None:
        s = "Number of steps required (flag -ns)"
        error(s)
    if int(ns) < 1:
        s = "Number of steps must be at least 1"
        error(s)
    return ns