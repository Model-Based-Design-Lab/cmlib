""" miscellaneous utility functions """

from math import gcd
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


def matrixFloatToFraction(M: Iterable[Iterable[float]])->List[List[Fraction]]:
    return [[Fraction(e) for e in r] for r in M]

def lcm(a: int, b: int)->int:
    '''Least common multiple (does not exist in math library python version <=3.8)'''
    return abs(a*b) // gcd(a, b)

def commonDenominator(den: int, v: Fraction)->int:
    return lcm(den,v.denominator)

def commonDenominatorList(l: List[Fraction])->int:
    den: int = 1
    for v in l:
        den = commonDenominator(den, v)
    return den

def primeFactors(n: int)->List[int]:
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

COMPLEXITY_THRESHOLD: int = 42

def complexity(n: int):
    primeFactorList = primeFactors(n)
    if len(primeFactorList) == 0:
        return 1
    return len(primeFactorList)*max(primeFactorList)

def isComplex(n: int)->bool:
    if n>1024:
        return True
    return complexity(n)>COMPLEXITY_THRESHOLD

def prettyPrintMatrix(M: List[List[Fraction]]):
    '''Print matrix M to the console.'''
    # get common denominator
    den: int = 1
    for r in M:
        den = lcm(den, commonDenominatorList(r))

    if isComplex(den):
        printMatrix(M)
    else:
        printFractionMatrix(M)

def determineMaxExp(M):
    mo = maxOpt([maxOpt([exponent(e) for e in r]) for r in M])
    if mo is None:
        return 0
    return mo

def determineMinExp(M):
    mo = minOpt([minOpt([exponent(e) for e in r]) for r in M])
    if mo is None:
        return 0
    return mo

def printMatrix(M: List[List[Fraction]]):
    '''Print matrix M to the console.'''
    if len(M) == 0:
        print('[[]]')
        return
    maxE: int = determineMaxExp(M)
    minE: int = determineMinExp(M)
    if maxE > 3 or maxE < -2:
        printMatrixWithExponent(M, maxE)
        return

    w: Optional[int] = determineMaxWidth(M)
    for i, v in enumerate(M):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(vectorToString(v, w), end='')
        if i < len(M)-1:
            print('')
        else:
            print(']')
