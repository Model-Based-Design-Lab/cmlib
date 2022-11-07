# numerical comparisons

from fractions import Fraction

NumericalEpsilon = 1e-8

class MPAlgebraException(Exception):
    pass

def significantlySmaller(x, y):
    return y-x > NumericalEpsilon 


MP_MINUSINFINITY = None

def MP_MAX(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return max(x, y)

def MP_PLUS(x, y):
    if x is None:
        return None
    if y is None:
        return None
    return Fraction(x+y).limit_denominator()

def MP_MINUS(x, y):
    if y == None:
        raise MPAlgebraException('Cannot subtract minus infinity')
    if x is None:
        return None
    return Fraction(x-y).limit_denominator()

def MP_LARGER(x, y):
    if x is None:
        return False
    if y is None:
        return True
    return significantlySmaller(y, x)
