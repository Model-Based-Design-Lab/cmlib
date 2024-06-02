"""
Linear Algebra support to solve Markov Chain problems.
Matrices are represented as a list of column vectors
"""

from fractions import Fraction
from functools import reduce
from typing import Callable, Dict, List

from markovchains.utils.utils import MarkovChainException

TVector = List[Fraction]
TMatrix = List[TVector] # a list of column(!)-vectors

TFloatVector = List[float]
TFloatMatrix = List[TFloatVector] # a list of column(!)-vectors

def zero_vector(n: int)->TVector:
    '''Generate a zero-vector'''
    return [Fraction(0) for _ in range(n)]

def one_vector(n: int)->TVector:
    '''Generate a one-vector'''
    return [Fraction(1) for _ in range(n)]

def zero_matrix(nr: int, nc: int)->TMatrix:
    '''Generate a zero-matrix'''
    return [zero_vector(nr) for _ in range(nc)]

def one_matrix(nr: int, nc: int)->TMatrix:
    '''Generate a one-matrix'''
    return [one_vector(nr) for _ in range(nc)]

def copy_vector(v: TVector)->TVector:
    """Create a copy of a vector."""
    return list(v)

def copy_matrix(a: TMatrix)->TMatrix:
    """Create a copy of a vector."""
    return [copy_vector(v) for v in a]

def inner_product(va: TVector, vb: TVector)->Fraction:
    '''compute inner product of vectors.'''
    return reduce(lambda sum,i: sum + va[i]*vb[i], range(len(va)), Fraction(0))

def matrix_vector_product(a: TMatrix, v: TVector)-> TVector:
    """Compute matrix-vector product."""
    n = len(v)
    return [
        reduce(lambda sum, c: sum+a[c][r]*v[c], range(n), Fraction(0)) for r in range(n) # ... it seems fine to me... pylint: disable=cell-var-from-loop
        ]

def vector_matrix_product(v: TVector, a: TMatrix)-> TVector:
    '''Compute the product between vector and matrix'''
    return [inner_product(v, vc) for vc in a]

def matrix_matrix_product(ma: TMatrix, mb: TMatrix)-> TMatrix:
    '''Compute the product of two matrices'''
    return [matrix_vector_product(ma, v) for v in mb]

def vector_sum(v: TVector)-> Fraction:
    '''Compute the sum of the vector of fractions.'''
    return reduce(lambda sum, x: sum+x, v, Fraction(0))

def fl_vector_sum(v: TFloatVector)-> float:
    '''Compute the sum of a vector of floats.'''
    return reduce(lambda sum, x: sum+x, v, 0.0)

def row_sum(ma: TMatrix)-> TVector:
    '''Sum matrix A along its rows'''
    if len(ma)==0:
        return []
    return [reduce(lambda sum, c: sum+ma[c][r], range(len(ma)) , Fraction(0)) for r in \
            range(len(ma[0]))] # ... it seems fine to me... pylint: disable=cell-var-from-loop

def column_sum(ma: TMatrix)-> TVector:
    '''Sum matrix A along its columns'''
    return [vector_sum(v) for v in ma]

def unit_vector(n: int, k: int)->TVector:
    '''Return unit vector k of length n.'''
    res = zero_vector(n)
    res[k] = Fraction(1)
    return res

def identity_matrix(n: int)->TMatrix:
    '''Return a n-by-n identity matrix.'''
    return [unit_vector(n, k) for k in range(n)]

def subtract_vector(va: TVector, vb: TVector)->TVector:
    '''Subtract vectors.'''
    return [va[k]-vb[k] for k in range(len(va))]

def fl_subtract_vector(va: TFloatVector, vb: TFloatVector)->TFloatVector:
    '''Subtract vectors of floats.'''
    return [va[k]-vb[k] for k in range(len(va))]

def add_vector(va: TVector, vb: TVector)->TVector:
    '''Add vectors.'''
    return [va[k]+vb[k] for k in range(len(va))]

def fl_add_vector(va: TFloatVector, vb: TFloatVector)->TFloatVector:
    '''Add vectors of floats.'''
    return [va[k]+vb[k] for k in range(len(va))]

def subtract_matrix(ma: TMatrix, mb: TMatrix)->TMatrix:
    '''Subtract two matrices.'''
    return [subtract_vector(ma[k], mb[k]) for k in range(len(ma))]

def add_matrix(ma: TMatrix, mb: TMatrix)->TMatrix:
    '''Add two matrices.'''
    return [add_vector(ma[k], mb[k]) for k in range(len(ma))]

def transpose(ma: TMatrix) -> TMatrix:
    '''Transpose matrix.'''
    if len(ma)==0:
        return [[]]
    n_rows_ma = len(ma[0])
    n_cols_ma = len(ma)
    return [[ma[c][r] for c in range(n_cols_ma)] for r in range(n_rows_ma)]

def fl_vector_element_wise(v: TFloatVector, f: Callable[[float],float ])->TFloatVector:
    '''Apply the given function to the elements of a vector to compute a new vector.'''
    return [f(x) for x in v]

def fl_matrix_element_wise(ma: TFloatMatrix, f: Callable[[float],float ])->TFloatMatrix:
    '''Apply the given function to the elements of a matrix to compute a new matrix.'''
    return [fl_vector_element_wise(v, f) for v in ma]

def mat_power(ma: TMatrix, n: int)->TMatrix:
    '''Raise matrix ma to the power n'''
    # if n is 0, or 1
    if n==0:
        return identity_matrix(len(ma))
    if n==1:
        return ma
    if n % 2 == 0:
        # if n is even A^n = (A^2)^n
        return mat_power(matrix_matrix_product(ma, ma), n//2)
    # if n=2m+1, A^n = (A^2)^m A
    return matrix_matrix_product(mat_power(matrix_matrix_product(ma, ma), n//2), ma)

def solve(ma: TMatrix, b: TVector)->TVector:
    '''Solve the linear equation Ax=b for a square and full rank matrix A'''
    # copy matrix and vector as we are going to change them
    ma = copy_matrix(ma)
    b = copy_vector(b)

    # Gaussian elimination from the top of my head..."
    a_size = len(ma)
    # map for indirect manipulation of A' w.r.t original A
    # A'[c][r] = A[c][rowIndex[r]]
    row_index: Dict[int,int] = {}
    for r in range(a_size):
        row_index[r]=r
    for r in range(a_size):
        # find a row k >= r s.t. A'[k][r] is not 0
        k = r
        while ma[r][row_index[k]] == Fraction(0):
            k = k + 1
            if k==a_size:
                raise MarkovChainException("Matrix is not full rank.")
        # In A', swap rows r and k
        row_index[k], row_index[r] = row_index[r], row_index[k]

        for rp in range(a_size):
            if r != rp:
                # replace row rp by rp-(rp[r]/r[r])r
                f: Fraction = ma[r][row_index[rp]] / ma[r][row_index[r]]
                b[row_index[rp]] = b[row_index[rp]] - f*b[row_index[r]]
                for c in range(a_size):
                    ma[c][row_index[rp]] = ma[c][row_index[rp]] - f*ma[c][row_index[r]]
            else:
                # replace row r by r * 1/ r[r]
                f: Fraction = Fraction(1) / ma[r][row_index[r]]
                b[row_index[r]] = f * b[row_index[r]]
                for c in range(a_size):
                    ma[c][row_index[r]] = f * ma[c][row_index[r]]

    # Equation is reduced to Ix=b'
    return [b[row_index[r]] for r in range(a_size)]

def invert_matrix(ma: TMatrix)->TMatrix:
    '''Return the inverse of the square and full rank matrix A'''
    # copy matrix and vector as we are going to change them
    ma = copy_matrix(ma)

    # Gaussian elimination from the top of my head..."
    a_size = len(ma)

    mb = identity_matrix(a_size)

    # map for indirect manipulation of A' w.r.t original A
    # A'[c][r] = A[c][rowIndex[r]]
    row_index = {}
    for r in range(a_size):
        row_index[r]=r
    for r in range(a_size):
        # find a row k >= r s.t. A'[k][r] is not 0
        k = r
        while ma[r][row_index[k]] == Fraction(0):
            k = k + 1
            if k==a_size:
                raise MarkovChainException("Matrix is not full rank.")
        # In A', swap rows r and k
        row_index[k], row_index[r] = row_index[r], row_index[k]

        for rp in range(a_size):
            if r != rp:
                # replace row rp by rp-(rp[r]/r[r])r
                f: Fraction = ma[r][row_index[rp]] / ma[r][row_index[r]]
                for c in range(a_size):
                    ma[c][row_index[rp]] = ma[c][row_index[rp]] - f*ma[c][row_index[r]]
                    mb[c][row_index[rp]] = mb[c][row_index[rp]] - f*mb[c][row_index[r]]
            else:
                # replace row r by r * 1/ r[r]
                f: Fraction = Fraction(1) / ma[r][row_index[r]]
                for c in range(a_size):
                    ma[c][row_index[r]] = f * ma[c][row_index[r]]
                    mb[c][row_index[r]] = f * mb[c][row_index[r]]

    # Equation is reduced to IX=B'=A^-1
    return [[mb[c][row_index[r]] for r in range(a_size)] for c in range(a_size)]
