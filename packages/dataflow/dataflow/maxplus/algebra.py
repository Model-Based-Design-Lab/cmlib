# numerical comparisons

from fractions import Fraction
from dataflow.maxplus.types import TTimeStamp


NumericalEpsilon = 1e-8

class MPAlgebraException(Exception):
    pass

def significantlySmaller(x: float, y: float)->bool:
    '''Test if x is significantly smaller than y.'''
    return y-x > NumericalEpsilon 

def significantlyLarger(x: float, y: float)->bool:
    '''Test if x is significantly larger than y.'''
    return significantlySmaller(y, x)


def approximatelyEqual(x: float, y: float)->bool:
    return (y-x < NumericalEpsilon) and (x-y < NumericalEpsilon)

MP_MINUSINFINITY = None

def MP_MAX(x: TTimeStamp, y: TTimeStamp) -> TTimeStamp:
    if x is None:
        return y
    if y is None:
        return x
    return max(x, y)

def MP_PLUS(x: TTimeStamp, y: TTimeStamp) -> TTimeStamp:
    if x is None:
        return None
    if y is None:
        return None
    return Fraction(x+y).limit_denominator()

def MP_MINUS(x: TTimeStamp, y: TTimeStamp)-> TTimeStamp:
    if y == None:
        raise MPAlgebraException('Cannot subtract minus infinity')
    if x is None:
        return None
    return Fraction(x-y).limit_denominator()

def MP_LARGER(x: TTimeStamp, y: TTimeStamp)->bool:
    if x is None:
        return False
    if y is None:
        return True
    return significantlySmaller(y, x)
