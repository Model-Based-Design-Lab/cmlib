from fractions import Fraction
from functools import reduce
from typing import Any, Callable, Dict, List

TVector = List[Fraction]
TMatrix = List[TVector] # a list of column(!)-vectors 

TFloatVector = List[float]
TFloatMatrix = List[TFloatVector] # a list of column(!)-vectors 

def zeroVector(n: int)->TVector:
    '''Generate a zero-vector'''
    return [Fraction(0) for _ in range(n)]

def oneVector(n: int)->TVector:
    '''Generate a one-vector'''
    return [Fraction(1) for _ in range(n)]

def zeroMatrix(nr: int, nc: int)->TMatrix:
    '''Generate a zero-matrix'''
    return [zeroVector(nr) for _ in range(nc)]

def oneMatrix(nr: int, nc: int)->TMatrix:
    '''Generate a one-matrix'''
    return [oneVector(nr) for _ in range(nc)]

def copyVector(v: TVector)->TVector:
    return [x for x in v]

def copyMatrix(A: TMatrix)->TMatrix:
    return [copyVector(v) for v in A]

def innerProduct(va: TVector, vb: TVector)->Fraction:
    '''compute inner product of vectors.'''
    return reduce(lambda sum,i: sum + va[i]*vb[i], range(len(va)), Fraction(0))

def matrixVectorProduct(A: TMatrix, v: TVector)-> TVector:
    N = len(v)
    return [reduce(lambda sum, c: sum+A[c][r]*v[c], range(N), Fraction(0)) for r in range(N)]

def vectorMatrixProduct(v: TVector, A: TMatrix)-> TVector:
    return [innerProduct(v, vc) for vc in A]

def matrixMatrixProduct(A: TMatrix, B: TMatrix)-> TMatrix:
    return [matrixVectorProduct(A, v) for v in B]

def vectorSum(v: TVector)-> Fraction:
    return reduce(lambda sum, x: sum+x, v, Fraction(0))

def flVectorSum(v: TFloatVector)-> float:
    return reduce(lambda sum, x: sum+x, v, 0.0)

def rowSum(A: TMatrix)-> TVector:
    '''Sum matrix A along its rows'''
    if len(A)==0:
        return []
    return [reduce(lambda sum, c: sum+A[c][r], range(len(A)) , Fraction(0)) for r in range(len(A[1]))]

def columnSum(A: TMatrix)-> TVector:
    '''Sum matrix A along its columns'''
    return [vectorSum(v) for v in A]


def unitVector(N: int, k: int)->TVector:
    res = zeroVector(N)
    res[k] = Fraction(1)
    return res

def identityMatrix(N: int)->TMatrix:
    return [unitVector(N, k) for k in range(N)]

def subtractVector(va: TVector, vb: TVector)->TVector:
    return [va[k]-vb[k] for k in range(len(va))]

def addVector(va: TVector, vb: TVector)->TVector:
    return [va[k]+vb[k] for k in range(len(va))]

def flAddVector(va: TFloatVector, vb: TFloatVector)->TFloatVector:
    return [va[k]+vb[k] for k in range(len(va))]

def subtractMatrix(Aa: TMatrix, Ab: TMatrix)->TMatrix:
    return [subtractVector(Aa[k], Ab[k]) for k in range(len(Aa))]

def addMatrix(Aa: TMatrix, Ab: TMatrix)->TMatrix:
    return [addVector(Aa[k], Ab[k]) for k in range(len(Aa))]

def transpose(A: TMatrix) -> TMatrix:
    if len(A)==0:
        return [[]]
    nRowsA = len(A[1])
    nColsA = len(A)
    return [[A[c][r] for c in range(nColsA)] for r in range(nRowsA)]

def flVectorElementwise(v: TFloatVector, f: Callable[[float],float ])->TFloatVector:
    return [f(x) for x in v]

def flMatrixElementwise(A: TFloatMatrix, f: Callable[[float],float ])->TFloatMatrix:
    return [flVectorElementwise(v, f) for v in A]

def matPower(A: TMatrix, n: int)->TMatrix:
    # if n is 0, or 1
    if n==0:
        return identityMatrix(len(A))
    if n==1:
        return A
    if n % 2 == 0:
        # if n is even A^n = (A^2)^n
        return matPower(matrixMatrixProduct(A, A), n//2)
    else:
        # if n=2m+1, A^n = (A^2)^m A
        return matrixMatrixProduct(matPower(matrixMatrixProduct(A, A), n//2), A)

def solve(A: TMatrix, b: TVector)->TVector:
    '''Solve the linear equation Ax=b for a square and full rank matrix A'''
    # copy matrix and vector as we are going to change them
    A = copyMatrix(A)
    b = copyVector(b)
    
    # Gaussian elimination from the top of my head..."
    AS = len(A)
    # map for indirect manipulation of A' w.r.t original A
    # A'[c][r] = A[c][rowIndex[r]]
    rowIndex: Dict[int,int] = dict()
    for r in range(AS):
        rowIndex[r]=r 
    for r in range(AS):
        # find a row k >= r s.t. A'[k][r] is not 0
        k = r
        while A[r][rowIndex[k]] == Fraction(0):
            k = k + 1
            if k==AS:
                raise Exception("Matrix is not full rank.")
        # In A', swap rows r and k
        rowIndex[k], rowIndex[r] = rowIndex[r], rowIndex[k] 

        for rp in range(AS):
            if r != rp:
                # replace row rp by rp-(rp[r]/r[r])r
                f: Fraction = A[r][rowIndex[rp]] / A[r][rowIndex[r]]
                b[rowIndex[rp]] = b[rowIndex[rp]] - f*b[rowIndex[r]]
                for c in range(AS):
                    A[c][rowIndex[rp]] = A[c][rowIndex[rp]] - f*A[c][rowIndex[r]]
            else:
                # replace row r by r * 1/ r[r]
                f: Fraction = Fraction(1) / A[r][rowIndex[r]]
                b[rowIndex[r]] = f * b[rowIndex[r]]
                for c in range(AS):
                    A[c][rowIndex[r]] = f * A[c][rowIndex[r]]

    # Equation is reduced to Ix=b'
    return [b[rowIndex[r]] for r in range(AS)]

def invertMatrix(A: TMatrix)->TMatrix:
    '''Return the inverse of the square and full rank matrix A'''
    # copy matrix and vector as we are going to change them
    A = copyMatrix(A)

    # Gaussian elimination from the top of my head..."
    AS = len(A)
    
    B = identityMatrix(AS)

    # map for indirect manipulation of A' w.r.t original A
    # A'[c][r] = A[c][rowIndex[r]]
    rowIndex = dict()
    for r in range(AS):
        rowIndex[r]=r 
    for r in range(AS):
        # find a row k >= r s.t. A'[k][r] is not 0
        k = r
        while A[r][rowIndex[k]] == Fraction(0):
            k = k + 1
            if k==AS:
                raise Exception("Matrix is not full rank.")
        # In A', swap rows r and k
        rowIndex[k], rowIndex[r] = rowIndex[r], rowIndex[k] 
            
        for rp in range(AS):
            if r != rp:
                # replace row rp by rp-(rp[r]/r[r])r
                f: Fraction = A[r][rowIndex[rp]] / A[r][rowIndex[r]]
                for c in range(AS):
                    A[c][rowIndex[rp]] = A[c][rowIndex[rp]] - f*A[c][rowIndex[r]]
                    B[c][rowIndex[rp]] = B[c][rowIndex[rp]] - f*B[c][rowIndex[r]]
            else:
                # replace row r by r * 1/ r[r]
                f: Fraction = Fraction(1) / A[r][rowIndex[r]]
                for c in range(AS):
                    A[c][rowIndex[r]] = f * A[c][rowIndex[r]]
                    B[c][rowIndex[r]] = f * B[c][rowIndex[r]]

    # Equation is reduced to IX=B'=A^-1
    return [[B[c][rowIndex[r]] for r in range(AS)] for c in range(AS)]
