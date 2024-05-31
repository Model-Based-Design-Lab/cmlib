""" miscellaneous utility functions """

from functools import reduce
from math import floor, gcd, log
from string import digits
import sys
import time
from fractions import Fraction
from typing import Callable, Iterable, List, Optional, Tuple, Union
import markovchains.utils.linalgebra as linalg
from markovchains.utils.statistics import StopConditions

NUM_FORMAT = '{:.5f}'
NUM_SCIENTIFIC = '{:.5e}'
COMPLEXITY_THRESHOLD: int = 50

class MarkovChainException(Exception):
    """Exceptions related to this package"""


# Frac = lambda n : Fraction(n).limit_denominator(max_denominator=1000000) if not np.isnan(n) else n

def warn(s: str):
    """Print warning."""
    print("Warning: " + s)

def error(s: str):
    """Print error."""
    print("Error: "+ s)
    sys.exit()

def only_digits(s: str)->str:
    """Filter only digits from string."""
    return ''.join(c for c in s if c in digits)

def only_non_digits(s: str)->str:
    """Filter only non-digits from string."""
    return ''.join(c for c in s if c not in digits)


def get_index(name: str)->int:
    """Get index from string."""
    dig = only_digits(name)
    if dig == '':
        return -1
    return int(dig)

def is_numerical(set_of_names: Iterable[str])->bool:
    """Check if the names are numbered with an identical based."""
    alpha_names = set([only_non_digits(s) for s in set_of_names])
    return len(alpha_names) <= 1

def string_to_float(s: str, default: float)->float:
    """Convert string to float."""
    try:
        return float(s)
    except ValueError:
        return default

def sort_names(set_of_names: Iterable[str])->List[str]:
    """Sort the set of names by number if possible, otherwise alphabetical."""

    list_of_names = list(set_of_names)
    if is_numerical(set_of_names):
        list_of_names.sort(key=get_index)
    else:
        list_of_names.sort()
    return list_of_names

def print_list(l: List[float])->str:
    """Print list to string."""
    try:
        string = "["
        for item in l:
            string += NUM_FORMAT.format(item)+", "
        return string[:-2] + "]"
    except ValueError:
        return str(l)

def print_optional_list(l: Optional[List[float]], none_text: str = "--")->str:
    """Print list of optional floats to string."""
    if l is None:
        return none_text
    return print_list(l)



def print_interval(i: Tuple[float,float])->str:
    """Print interval to string."""
    return print_list([i[0],i[1]])

def print_list_of_intervals(l: List[Tuple[float,float]])->str:
    """Print list of intervals to string."""
    try:
        string = "["
        for i in l:
            string += print_interval(i) + ", "
        return string[:-2] + "]"
    except ValueError:
        return str(l)

def print_optional_interval(i: Optional[Tuple[float,float]])->str:
    """Print optional interval to string."""
    if i is None:
        return "--"
    return print_list([i[0],i[1]])

def print_optional_list_of_intervals(l: Optional[List[Tuple[float,float]]])->str:
    """Print optional list of intervals to string."""
    if l is None:
        return "--"
    return print_list_of_intervals(l)

def optional_float_or_string_to_string(nr: Optional[Union[str,float]]):
    """Print an optional float or string to a string"""
    if nr is None:
        return "--"
    if isinstance(nr, str):
        return nr
    try:
        return NUM_FORMAT.format(nr)
    except ValueError:
        return f"{nr}"


def print_sorted_set(s: Iterable[str]):
    """Print a set sorted to comma separated list."""
    print(f"{{{', '.join(sort_names(s))}}}")

def print_sorted_list(s: Iterable[str]):
    """Print a list, sorted, to comma separated list."""
    print(f"{', '.join(sort_names(s))}")

def print_list_of_strings(s: List[str]):
    """Print list of strings comma separated."""
    print (f"[{', '.join(s)}]\n")

def print_table(table: List[Union[str,List[str]]], margin: int = 1):
    """Print a table of strings to screen."""

    # determine nr of columns of table
    nr_columns: int = -1
    for row in table:
        if isinstance(row, list):
            nr_columns = len(row)

    if nr_columns == -1:
        # table has no proper rows
        for row in table:
            print(row)
            return

    # determine column widths
    widths: List[int] = []
    for column in range(nr_columns):
        w: int = 0
        for row in table:
            if isinstance(row, list):
                w = max(w, len(row[column]))
        widths.append(w)

    # print the table
    for row in table:
        if isinstance(row, str):
            print(row)
        else:
            for i, v in enumerate(row):
                if i > 0:
                    print(" ".ljust(margin), end="")
                print(v.ljust(widths[i]), end="")
            print()

def stop_criteria(c: List[float])->StopConditions:
    '''Create stop conditions from list of floats.'''
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

    return StopConditions(c[0],c[1], c[2], int(c[3]), int(c[4]), c[5])

def nr_of_steps(ns: int)->int:
    """Extract and validate number of steps."""
    if ns is None:
        s = "Number of steps required (flag -ns)"
        error(s)
    if int(ns) < 1:
        s = "Number of steps must be at least 1"
        error(s)
    return ns

def vector_float_to_fraction(v: Iterable[float])->List[Fraction]:
    """Convert vector of floats to vector of fractions."""
    return [Fraction(e).limit_denominator(10000) for e in v]

def matrix_float_to_fraction(matrix: Iterable[Iterable[float]])->List[List[Fraction]]:
    """Convert matrix of floats to matrix of fractions."""
    return [[Fraction(e).limit_denominator(10000) for e in r] for r in matrix]

def lcm(a: int, b: int)->int:
    '''Least common multiple (does not exist in math library python version <=3.8)'''
    return abs(a*b) // gcd(a, b)

def common_denominator(den: int, v: Fraction)->int:
    """Determine common denominator between denominator and fraction."""
    return lcm(den,v.denominator)

def common_denominator_list(l: List[Fraction])->int:
    """Determine the common denominator for a list of fractions."""
    den: int = 1
    for v in l:
        den = common_denominator(den, v)
    return den

def prime_factors(n: int)->List[int]:
    """Break a number in its prime factors."""
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

def complexity(n: int):
    """Determine a metric of complexity of a number as the product of the number of
    its distinct prime factors and the largest prime factor."""
    prime_factor_list = prime_factors(n)
    if len(prime_factor_list) == 0:
        return 1
    return len(prime_factor_list)*max(prime_factor_list)

def is_complex(n: int)->bool:
    """Determine if a given integer is 'complex' or not (to decide between representation
    as a fraction or a float)"""
    return complexity(n)>COMPLEXITY_THRESHOLD

def vector_fraction_to_float(v: List[Fraction])->List[float]:
    """Convert list of fractions to list of floats."""
    return [float(e) for e in v]

def matrix_fraction_to_float(matrix: List[List[Fraction]])->List[List[float]]:
    """Convert matrix of fractions to matrix of floats."""
    return [[float(e) for e in r] for r in matrix]

def pretty_print_matrix(matrix: List[List[Fraction]]):
    '''Print matrix M to the console.'''

    # transform to row major for easy printing
    matrix = linalg.transpose(matrix)

    # get common denominator
    den: int = 1
    for r in matrix:
        den = lcm(den, common_denominator_list(r))
    if is_complex(den):
        print_matrix(matrix_fraction_to_float(matrix))
    else:
        print_fraction_matrix(matrix)

def pretty_print_vector(v: List[Fraction]):
    """Pretty print a vector of fractions to screen."""
    # get common denominator
    den = common_denominator_list(v)
    if is_complex(den):
        print_vector(vector_fraction_to_float(v))
    else:
        print_fraction_vector(v)

def pretty_print_value(x: Fraction, end=None):
    """Pretty print a fraction value to screen."""
    if is_complex(x.denominator):
        print(NUM_FORMAT.format(x), end=end)
    else:
        print(x, end=end)

def max_opt(l: List[Optional[int]])-> Optional[int]:
    """Determine the maximum of a list of optional integers."""
    return reduce(lambda m, v: v if m is None else (m if v is None else max(v,m)), l, None)

def min_opt(l: List[Optional[int]])-> Optional[int]:
    """Determine the minimum of a list of optional integers."""
    return reduce(lambda m, v: v if m is None else (m if v is None else min(v,m)), l, None)

def exponent(e: Optional[float])->Optional[int]:
    """Determine the order of magnitude of an optional float."""
    if e is None:
        return None
    if e==0.0:
        return None
    return floor(log(abs(e), 10.0))


def determine_max_exp(matrix: List[List[float]]):
    """Determine the maximum order of magnitude of the values in a matrix."""
    mo = max_opt([max_opt([exponent(e) for e in r]) for r in matrix])
    if mo is None:
        return 0
    return mo

def determine_max_exp_vector(v: List[float]):
    """Determine the maximum order of magnitude of the values in a vector."""
    mo = max_opt([exponent(e) for e in v])
    if mo is None:
        return 0
    return mo

def determine_min_exp(matrix: List[List[float]]):
    """Determine the minimum order of magnitude of the values in a matrix."""
    mo = min_opt([min_opt([exponent(e) for e in r]) for r in matrix])
    if mo is None:
        return 0
    return mo

def exp_matrix(matrix: List[List[float]], ex: int) -> List[List[float]]:
    """Scale the values in the matrix by 10 to the power ex."""
    f = pow(10,ex)
    return [[e * f for e in r] for r in matrix]

def exp_vector(v: List[float], ex: int) -> List[float]:
    """Scale the values in the vector by 10 to the power ex."""
    f = pow(10,ex)
    return [e * f for e in v]

def element_to_string(x: float, w: Optional[int]=None)->str:
    """Convert float to a string of width w choosing optimal scientific or normal representation."""
    ex = 0 if x==0.0 else log(abs(x),10)
    fmt = NUM_FORMAT if -3 <= ex <= 5 else NUM_SCIENTIFIC
    return right_align(fmt.format(float(x)),w) if w else fmt.format(float(x))

def right_align(s: str, w: int)->str:
    """Right align and extend string to length w."""
    return (' '*(w-len(s)))+s

def determine_fraction_width(e: Fraction)->int:
    """Determine the string width of the given fraction."""
    return len(f'{e}')

def determine_max_fraction_width(matrix: List[List[Fraction]])->Optional[int]:
    """Determine the maximum width of the fractions in the matrix."""
    return max_opt([determine_max_faction_width_vector(r) for r in matrix])

def determine_max_faction_width_vector(v: List[Fraction])->Optional[int]:
    """Determine the maximum width of the fractions in the vector."""
    return max_opt([determine_fraction_width(e) for e in v])

def determine_width(e: float)->int:
    """Determine the width of the float represented as a string."""
    return len(element_to_string(e))

def determine_max_width(matrix: List[List[float]])->Optional[int]:
    """Determine the maximum width of the floats in the matrix represented as strings."""
    return max_opt([determine_max_width_vector(r) for r in matrix])

def determine_max_width_vector(v: List[float])->Optional[int]:
    """Determine the maximum width of the floats in the vector represented as strings."""
    return max_opt([determine_width(e) for e in v])

def vector_to_string(v: List[float], w: Optional[int]=None)->str:
    '''Return string representation of the vector v.'''
    if w is None:
        w = determine_max_width_vector(v)
    return f"[ {' '.join([element_to_string(x,w) for x in v])} ]"

def print_matrix_with_exponent(matrix: List[List[float]], ex: int):
    """Print matrix with factored exponent."""

    matrix = exp_matrix(matrix, -ex)

    exp_prefix = f'10^{ex} x '
    spc_prefix = ' ' * (len(exp_prefix)+1)

    w: Optional[int] = determine_max_width(matrix)
    for i, v in enumerate(matrix):
        if i == 0:
            print(exp_prefix+'[', end="")
        else:
            print(spc_prefix, end="")
        print(vector_to_string(v, w), end='')
        if i < len(matrix)-1:
            print('')
        else:
            print(']')

def print_vector_with_exponent(v: List[float], ex: int):
    """Print a vector with factored exponent."""

    v = exp_vector(v, -ex)

    exp_prefix = '10^{ex} x '

    w: Optional[int] = determine_max_width_vector(v)
    print(exp_prefix, end="")
    print(vector_to_string(v, w), end='')

def print_vector(v: List[float]):
    """Print vector to screen."""
    if len(v) == 0:
        print('[]')
        return
    max_e: int = determine_max_exp_vector(v)
    if max_e > 3 or max_e < -2:
        print_vector_with_exponent(v, max_e)
        return

    w: Optional[int] = determine_max_width_vector(v)
    print(vector_to_string(v, w))

def print_matrix(matrix: List[List[float]]):
    '''Print matrix M to the console.'''
    if len(matrix) == 0:
        print('[[]]')
        return
    max_e: int = determine_max_exp(matrix)
    if max_e > 3 or max_e < -2:
        print_matrix_with_exponent(matrix, max_e)
        return

    w: Optional[int] = determine_max_width(matrix)
    for i, v in enumerate(matrix):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(vector_to_string(v, w), end='')
        if i < len(matrix)-1:
            print('')
        else:
            print(']')

def element_to_fraction_string(x: Fraction, w: Optional[int]=None)->str:
    """Convert fraction to string at optional width."""
    return right_align(f'{x}', w) if w else f'{x}'

def vector_to_fraction_string(v: List[Fraction], w: Optional[int]=None)->str:
    '''Return string representation of the vector v.'''
    if w is None:
        w = determine_max_faction_width_vector(v)
    return f"[ {'  '.join([element_to_fraction_string(x, w) for x in v])} ]"

def print_fraction_matrix(matrix: List[List[Fraction]]):
    '''Print matrix M to the console.'''
    if len(matrix) == 0:
        print('[[]]')
        return
    w: Optional[int] = determine_max_fraction_width(matrix)
    for i, v in enumerate(matrix):
        if i == 0:
            print('[', end="")
        else:
            print(' ', end="")
        print(vector_to_fraction_string(v, w), end='')
        if i < len(matrix)-1:
            print('')
        else:
            print(']')

def print_fraction_vector(v: List[Fraction]):
    """Print vector of fractions."""
    if len(v) == 0:
        print('[]')
        return
    w: Optional[int] = determine_max_faction_width_vector(v)
    print(vector_to_fraction_string(v, w))

class TimeoutTimer:
    """Timeout timer for simulations with timeout."""

    _seconds_timeout: float
    _initial_time: float
    _active: bool

    def __init__(self, seconds_time_out: float) -> None:
        self._seconds_timeout = seconds_time_out
        self._active = seconds_time_out > 0.0
        self._initial_time = time.time()

    def is_expired(self)->bool:
        """Check if timer has expired."""
        if self._active:
            return self._seconds_timeout <= time.time() - self._initial_time
        return False

    def sim_action(self)-> Callable[[int,str],bool]:
        """Return a function that returns true if timer is active and expired."""
        if self._active:
            return lambda n, state: self._seconds_timeout <= time.time() - self._initial_time
        return lambda n, state: False
