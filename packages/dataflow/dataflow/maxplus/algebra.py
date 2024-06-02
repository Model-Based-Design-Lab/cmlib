'''max-plus operators and comparisons'''

from fractions import Fraction
from dataflow.maxplus.types import TTimeStamp


NUMERICAL_EPSILON = 1e-8

class MPAlgebraException(Exception):
    '''Exceptions related to max-plus algebra.'''

# def significantlySmaller(x: float, y: float)->bool:
#     '''Test if x is significantly smaller than y.'''
#     return y-x > NumericalEpsilon

# def significantlyLarger(x: float, y: float)->bool:
#     '''Test if x is significantly larger than y.'''
#     return significantlySmaller(y, x)


# def approximatelyEqual(x: float, y: float)->bool:
#     return (y-x < NumericalEpsilon) and (x-y < NumericalEpsilon)

MP_MINUSINFINITY_STR = "-inf"
MP_MINUSINFINITY = None

def mp_op_max(x: TTimeStamp, y: TTimeStamp) -> TTimeStamp:
    '''Max operation in max-plus algebra.'''
    if x is None:
        return y
    if y is None:
        return x
    return max(x, y)

def mp_op_plus(x: TTimeStamp, y: TTimeStamp) -> TTimeStamp:
    '''Plus operation in max-plus algebra.'''
    if x is None:
        return None
    if y is None:
        return None
    return Fraction(x+y).limit_denominator()

def mp_op_minus(x: TTimeStamp, y: TTimeStamp)-> TTimeStamp:
    '''Minus operation (inverse of plus) in max-plus algebra.'''
    if y is None:
        raise MPAlgebraException('Cannot subtract minus infinity')
    if x is None:
        return None
    return Fraction(x-y).limit_denominator()

def mp_comp_larger(x: TTimeStamp, y: TTimeStamp)->bool:
    '''Compare timestamp and check if x is strictly larger than y.'''
    if x is None:
        return False
    if y is None:
        return True
    return y<x
