'''Printing of max-plus related objects.'''

from fractions import Fraction
from functools import reduce
from math import floor, gcd, log
from typing import List, Optional, Union
from dataflow.maxplus.types import TMPMatrix, TMPVector, TMPVectorList, TTimeStamp
from dataflow.maxplus.algebra import MP_MINUSINFINITY, MP_MINUSINFINITY_STR


NUM_FORMAT = '{:.5f}'
NUM_SCIENTIFIC = '{:.5e}'

def lcm(a: int, b: int)->int:
    '''Least common multiple (does not exist in math library python version <=3.8)'''
    return abs(a*b) // gcd(a, b)

def mp_element_to_string(x: TTimeStamp, w: Optional[int]=None)->str:
    '''Convert time stamp to string.'''
    if x is MP_MINUSINFINITY:
        return right_align(MP_MINUSINFINITY_STR,w) if w else MP_MINUSINFINITY_STR
    ex = 0 if x==0.0 else log(abs(x),10) # type: ignore
    fmt = NUM_FORMAT if -3 <= ex <= 5 else NUM_SCIENTIFIC
    return right_align(fmt.format(float(x)),w) if w else fmt.format(float(x))  # type: ignore

def right_align(s: str, w: int)->str:
    '''Right align string to length w.'''
    return (' '*(w-len(s)))+s

def mp_element_to_fraction_string(x: TTimeStamp, w: Optional[int]=None)->str:
    '''Convert time stamp to fraction representation.g.'''
    if x is MP_MINUSINFINITY:
        return right_align(MP_MINUSINFINITY_STR, w) if w else MP_MINUSINFINITY_STR
    return right_align(f'{x}', w) if w else f'{x}'


def mp_vector_to_string(v: TMPVector, w: Optional[int]=None)->str:
    '''Return string representation of the vector v.'''
    if w is None:
        w = determine_max_width_vector(v)
    elements = ' '.join([mp_element_to_string(x,w) for x in v])
    return f'[ {elements} ]'

def mp_vector_to_fraction_string(v: TMPVector, w: Optional[int]=None)->str:
    '''Return string representation of the vector v.'''
    if w is None:
        w = determine_max_fraction_width_vector(v)
    elements = '  '.join([mp_element_to_fraction_string(x, w) for x in v])
    return f'[ {elements} ]'

def mp_pretty_vector_to_string(v: TMPVector)->str:
    '''Print vector to string, as fractions is nicer.'''
    # get common denominator
    den = common_denominator_list(v)
    if is_complex(den):
        return mp_vector_to_string(v)
    return mp_vector_to_fraction_string(v)

def mp_pretty_value(v: TTimeStamp)->str:
    '''Print timestamp to string, as fraction if nicer.'''
    # get common denominator
    den = 0
    if v is not None:
        den = v.denominator
    if is_complex(den):
        return mp_element_to_string(v)
    return mp_element_to_fraction_string(v)

def exponent(e: Union[TTimeStamp,Optional[float]])->Optional[int]:
    '''Determine order of magnitude.'''
    if e is None:
        return None
    if e==0.0:
        return None
    return floor(log(abs(e), 10.0))

def max_opt(l: List[Optional[int]])-> Optional[int]:
    '''Determine maximum of list of optional numbers.'''
    return reduce(lambda m, v: v if m is None else (m if v is None else max(v,m)), l, None)

def min_opt(l: List[Optional[int]])-> Optional[int]:
    '''Determine minimum of list of optional numbers.'''
    return reduce(lambda m, v: v if m is None else (m if v is None else min(v,m)), l, None)

def determine_max_exp(matrix: TMPMatrix):
    '''Determine the maximum order of magnitude in matrix M'''
    mo = max_opt([max_opt([exponent(e) for e in r]) for r in matrix])
    if mo is None:
        return 0
    return mo

def determine_min_exp(matrix: TMPMatrix):
    '''Determine the minimum order of magnitude in matrix M'''
    mo = min_opt([min_opt([exponent(e) for e in r]) for r in matrix])
    if mo is None:
        return 0
    return mo

def exp_matrix(matrix: TMPMatrix, ex: int) -> TMPMatrix:
    '''Factor out exponent from matrix.'''
    f = pow(10,ex)
    return [[None if e is None else e * f for e in r] for r in matrix]


def print_mp_matrix_with_exponent(matrix: TMPMatrix, ex: int):
    '''Print a matrix with factored out exponent.'''

    matrix = exp_matrix(matrix, -ex)

    exp_prefix = f'10^{ex} x '
    spc_prefix = ' ' * (len(exp_prefix)+1)

    w: Optional[int] = determine_max_width(matrix)
    for i, v in enumerate(matrix):
        if i == 0:
            print(exp_prefix+'[', end="")
        else:
            print(spc_prefix, end="")
        print(mp_vector_to_string(v, w), end='')
        if i < len(matrix)-1:
            print('')
        else:
            print(']')

def print_mp_matrix(matrix: TMPMatrix, nr: Optional[int]=None, nc: Optional[int]=None):
    '''Print matrix M to the console.'''
    if len(matrix) == 0:
        print_empty_matrix(nr, nc)
        return
    max_e: int = determine_max_exp(matrix)
    if max_e > 3 or max_e < -2:
        print_mp_matrix_with_exponent(matrix, max_e)
        return

    w: Optional[int] = determine_max_width(matrix)
    for i, v in enumerate(matrix):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(mp_vector_to_string(v, w), end='')
        if i < len(matrix)-1:
            print('')
        else:
            print(']')

def determine_fraction_width(e: TTimeStamp)->int:
    '''Determine width of the representation as a fraction.'''
    if e is MP_MINUSINFINITY:
        return len(MP_MINUSINFINITY_STR)
    return len(f'{e}')

def determine_width(e: TTimeStamp)->int:
    '''Determine width of the representation.'''
    return len(mp_element_to_string(e))

def print_empty_matrix(nr: Optional[int]=None, nc: Optional[int]=None):
    '''Print an empty matrix with the given number of rows and columns.'''
    empty_row='  '.join('[]'*nc) if nc else '[]'
    if not nr:
        print(f'[{empty_row}]')
    else:
        for n in range(nr):
            if n==0:
                print('[ ', end='')
            else:
                print('  ', end='')
            print(empty_row)
            if n==nr-1:
                print(']', end='')
        print()

def determine_max_fraction_width(matrix: TMPMatrix)->Optional[int]:
    '''Determine the maximum width of the representations of the elements of the
    matrix as fractions.'''
    return max_opt([determine_max_fraction_width_vector(r) for r in matrix])

def determine_max_fraction_width_vector(v: TMPVector)->Optional[int]:
    '''Determine the maximum width of the representations of the elements of the
    vector as fractions.'''
    return max_opt([determine_fraction_width(e) for e in v])

def determine_max_width(matrix: TMPMatrix)->Optional[int]:
    '''Determine the maximum width of the representations of the elements of the
    matrix.'''
    return max_opt([determine_max_width_vector(r) for r in matrix])

def determine_max_width_vector(v: TMPVector)->Optional[int]:
    '''Determine the maximum width of the representations of the elements of the
    vector.'''
    return max_opt([determine_width(e) for e in v])


def print_fraction_mp_matrix(matrix: TMPMatrix, nr: Optional[int]=None, nc: Optional[int]=None):
    '''Print matrix M to the console.'''
    if len(matrix) == 0:
        print_empty_matrix(nr, nc)
        return
    w: Optional[int] = determine_max_fraction_width(matrix)
    for i, v in enumerate(matrix):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(mp_vector_to_fraction_string(v, w), end='')
        if i < len(matrix)-1:
            print('')
        else:
            print(']')


def print_mp_vector_list(vl: TMPVectorList):
    '''Print list of vectors to the console.'''
    for i, v in enumerate(vl):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(mp_vector_to_string(v), end='')
        if i < len(vl)-1:
            print('')
        else:
            print(']')

def pretty_print_mp_matrix(matrix: TMPMatrix, nr: Optional[int]=None, nc: Optional[int]=None):
    '''Print matrix M to the console.'''
    # get common denominator
    den: int = 1
    for r in matrix:
        den = common_denominator_list(r, den)

    if is_complex(den):
        print_mp_matrix(matrix, nr, nc)
    else:
        print_fraction_mp_matrix(matrix, nr, nc)


# pretty printing, consider the maximum prime factor and the number of prime factors
# of the common denominator to determine the 'complexity' of fractions, before switching to decimal.

def common_denominator(den: int, v: Fraction)->int:
    '''Determine the common denominator between denominator den and the denominator of v.'''
    return lcm(den,v.denominator)

def common_denominator_list(l: List[Optional[Fraction]], den: int=1)->int:
    '''Determine the common denominator of a list of fractions.'''
    for v in l:
        if v is not None:
            den = common_denominator(den, v)
    return den

def common_denominator_matrix(l: List[List[Optional[Fraction]]])->int:
    '''Determine the common denominator of the elements of a matrix.'''
    den: int = 1
    for r in l:
        den = common_denominator_list(r, den)
    return den


def prime_factors(n: int)->List[int]:
    '''Determine the prime factors of a number.'''
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
    '''Determine a measure of complexity of a number, defined as the number of distinct prime
    factors multiplied by the largest prime factor.'''
    prime_factor_list = prime_factors(n)
    if len(prime_factor_list) == 0:
        return 1
    return len(prime_factor_list)*max(prime_factor_list)

def is_complex(n: int)->bool:
    '''Decide of a number is 'complex' or not.'''
    if n>1024:
        return True
    return complexity(n)>COMPLEXITY_THRESHOLD
