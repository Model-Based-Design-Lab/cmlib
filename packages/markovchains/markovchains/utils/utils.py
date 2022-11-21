""" miscellaneous utility functions """

from string import digits
from fractions import Fraction
import numpy as np

Frac = lambda n : Fraction(n).limit_denominator(max_denominator=1000000) if not np.isnan(n) else n
from typing import Iterable, List, Optional, Set, Tuple, Union

def warn(s: str):
    print("Warning: " + s)

def error(s: str):
    print("Error: "+ s)
    exit()

def onlyDigits(s: str)->str:
    return ''.join(c for c in s if c in digits)

def onlyNonDigits(s: str)->str:
    return ''.join(c for c in s if c not in digits)


def getIndex(name: str)->int:
    dig = onlyDigits(name)
    if dig == '':
        return -1
    else:
        return int(dig)

def isNumerical(setOfNames: Iterable[str])->bool:
    alphaNames = set([onlyNonDigits(s) for s in setOfNames])
    return len(alphaNames) <= 1

def stringToFloat(s: str, default: float)->float:
    try:
        return float(s)
    except ValueError:
        return default


def sortNames(setOfNames: Iterable[str])->List[str]:

    listOfNames = list(setOfNames)
    if isNumerical(setOfNames):
        listOfNames.sort(key=getIndex)
    else:
        listOfNames.sort()
    return listOfNames

def printList(l: List[float])->str:
    try:
        string = "["
        for item in l:
            string += "{:.4f}, ".format(item)
        return string[:-2] + "]"
    except:
        return str(l)

def printOptionalList(l: Optional[List[float]])->str:
    if l is None:
        return "--"
    return printList(l)



def printInterval(i: Tuple[float,float])->str:
    return printList([i[0],i[1]])

def printListOfIntervals(l: List[Tuple[float,float]])->str:
    try:
        string = "["
        for i in l:
            string += printInterval(i) + ", "
        return string[:-2] + "]"
    except:
        return str(l)



def printOptionalInterval(i: Optional[Tuple[float,float]])->str:
    if i is None:
        return "--"
    return printList([i[0],i[1]])

def printOptionalListOfIntervals(l: Optional[List[Tuple[float,float]]])->str:
    if l is None:
        return "--"
    return printListOfIntervals(l)
    
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



def printDList(dl:List[List[float]]):
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

def print4F(nr: float):
    try:
        return "{:.4f}".format(nr)
    except:
        return "{}".format(nr)

def printOptional4FOrString(nr: Optional[Union[str,float]]):
    if nr is None:
        return "--"
    if isinstance(nr, str):
        return nr
    try:
        return "{:.4f}".format(nr)
    except:
        return "{}".format(nr)


def printSortedSet(s: Iterable[str]):
    print("{{{}}}".format(", ".join(sortNames(s))))

def printSortedList(s: Iterable[str]):
    print("{}".format(", ".join(sortNames(s))))

def printVector(s: List[str]):
    print ("[{}]\n".format(", ".join(s)))


def stopCriteria(c: List[float])->List[float]:
    # Format stop criteria: 
    stop = '''
    Steady state behavior = 
    [
        Confidence level,
        Absolute error,
        Relative error,
        Maximum path length,
        Maximum number of cycles,
        Time (in seconds)
    ]

    Transient behavior = 
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

def nrOfSteps(ns: int)->int:
    if ns == None:
        s = "Number of steps required (flag -ns)"
        error(s)
    if int(ns) < 1:
        s = "Number of steps must be at least 1"
        error(s)
    return ns