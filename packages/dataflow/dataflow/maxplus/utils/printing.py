
from fractions import Fraction
from functools import reduce
from math import floor, lcm, log
from typing import List, Optional
from dataflow.maxplus.types import TMPMatrix, TMPVector, TMPVectorList, TTimeStamp
from dataflow.maxplus.algebra import MP_MINUSINFINITY

NUM_FORMAT = '{:.5f}'
NUM_SCIENTIFIC = '{:.5e}'

def mpElementToString(x: TTimeStamp, w: Optional[int]=None, miStr: str = '-inf')->str:
    if x is MP_MINUSINFINITY:
        return rightAlign(miStr,w) if w else miStr
    ex = 0 if x==0.0 else log(abs(x),10) # type: ignore
    fmt = NUM_FORMAT if -3 <= ex <= 5 else NUM_SCIENTIFIC 
    return rightAlign(fmt.format(float(x)),w) if w else fmt.format(float(x))  # type: ignore

def rightAlign(s: str, w: int)->str:
    return (' '*(w-len(s)))+s

def mpElementToFractionString(x: TTimeStamp, w: Optional[int]=None, miStr: str = '-inf')->str:
    '''Return a 6-character wide string representation of the max-plus element x, using miStr, defaulting to '-inf' to represent minus infinity.'''
    if x is MP_MINUSINFINITY:
        return rightAlign(miStr, w) if w else miStr
    return rightAlign('{}'.format(x), w) if w else '{}'.format(x)


def mpVectorToString(v: TMPVector, w: Optional[int]=None)->str:
    '''Return string representation of the vector v.'''
    return '[ {} ]'.format(' '.join([mpElementToString(x,w) for x in v]))

def mpVectorToFractionString(v: TMPVector, w: Optional[int]=None)->str:
    '''Return string representation of the vector v.'''
    return '[ {} ]'.format('  '.join([mpElementToFractionString(x, w) for x in v]))

def exponent(e: Optional[float])->Optional[int]:
    if e is None:
        return None
    else:
        if e==0.0:
            return None
        return floor(log(abs(e), 10.0))

def maxOpt(l: List[Optional[int]])-> Optional[int]:
    return reduce(lambda m, v: v if m is None else (m if v is None else max(v,m)), l, None)

def minOpt(l: List[Optional[int]])-> Optional[int]:
    return reduce(lambda m, v: v if m is None else (m if v is None else min(v,m)), l, None)

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

def expMatrix(M: TMPMatrix, ex: int) -> TMPMatrix:
    f = pow(10,ex)
    return [[None if e is None else e * f for e in r] for r in M]


def printMPMatrixWithExponent(M: TMPMatrix, ex: int):

    M = expMatrix(M, -ex)

    expPrefix = '10^{} x '.format(ex)
    spcPrefix = ' ' * (len(expPrefix)+1)

    w: Optional[int] = determineMaxWidth(M)
    for i, v in enumerate(M):
        if i == 0:
            print(expPrefix+'[', end="")
        else:
            print(spcPrefix, end="")
        print(mpVectorToString(v, w), end='')
        if i < len(M)-1:
            print('')
        else:
            print(']')

def printMPMatrix(M: TMPMatrix, nr: Optional[int]=None, nc: Optional[int]=None):
    '''Print matrix M to the console.'''
    if len(M) == 0:
        printEmptyMatrix(nr, nc)
        return
    maxE: int = determineMaxExp(M)
    minE: int = determineMinExp(M)
    if maxE > 3 or maxE < -2:
        printMPMatrixWithExponent(M, maxE)
        return

    w: Optional[int] = determineMaxWidth(M)
    for i, v in enumerate(M):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(mpVectorToString(v, w), end='')
        if i < len(M)-1:
            print('')
        else:
            print(']')

def determineFractionWidth(e: TTimeStamp, miStr: str = '-inf')->int:
    if e is MP_MINUSINFINITY:
        return len(miStr)
    return len('{}'.format(e))

def determineWidth(e: TTimeStamp, miStr: str = '-inf')->int:
    return len(mpElementToString(e))

def printEmptyMatrix(nr: Optional[int]=None, nc: Optional[int]=None):
    emptyRow='  '.join('[]'*nc) if nc else '[]'
    if not nr:
        print('[{}]'.format(emptyRow))
    else:
        for n in range(nr):
            if n==0:
                print('[ ', end='')
            else:
                print('  ', end='')
            print(emptyRow)
            if n==nr-1:
                print(']', end='')
        print()

def determineMaxFractionWidth(M: TMPMatrix)->Optional[int]:
    return maxOpt([maxOpt([determineFractionWidth(e) for e in r]) for r in M])

def determineMaxWidth(M: TMPMatrix)->Optional[int]:
    return maxOpt([maxOpt([determineWidth(e) for e in r]) for r in M])

def printFractionMPMatrix(M: TMPMatrix, nr: Optional[int]=None, nc: Optional[int]=None):
    '''Print matrix M to the console.'''
    if len(M) == 0:
        printEmptyMatrix(nr, nc)
        return
    w: Optional[int] = determineMaxFractionWidth(M)
    for i, v in enumerate(M):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(mpVectorToFractionString(v, w), end='')
        if i < len(M)-1:
            print('')
        else:
            print(']')


def printMPVectorList(vl: TMPVectorList):
    '''Print list of vectors to the console.'''
    for i, v in enumerate(vl):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(mpVectorToString(v), end='')
        if i < len(vl)-1:
            print('')
        else:
            print(']')

def prettyPrintMPMatrix(M: TMPMatrix, nr: Optional[int]=None, nc: Optional[int]=None):
    '''Print matrix M to the console.'''
    # get common denominator
    den: int = 1
    for r in M:
        den = lcm(den, commonDenominatorList(r))
    if isComplex(den):
        printMPMatrix(M, nr, nc)
    else:
        printFractionMPMatrix(M, nr, nc)

# pretty printing, consider the maximum prime factor and the number of prime factors (use their product?) of th common denominator to determine the 'complexity' of fractions, before switching to decimal.

def commonDenominator(den: int, v: Fraction)->int:
    return lcm(den,v.denominator)

def commonDenominatorList(l: List[Optional[Fraction]])->int:
    den: int = 1
    for v in l:
        if v is not None:
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
