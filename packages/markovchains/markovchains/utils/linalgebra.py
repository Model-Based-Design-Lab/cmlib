from fractions import Fraction
from typing import Any, List

def matPower(m: Any, n: int)->Any:
    return np.linalg.matrix_power(m, n)

def printMatrix(m):
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, linewidth=100000, threshold=100000)
    print(m)


def solve(A: List[List[Fraction]], b: List[Fraction]):
    '''Solve the linear equation Ax=b for a square and full rank matrix A'''
    # Gaussian elimination from the top of my head..."
    AS =len(A)
    
    # map for indirect manipulation of A' w.r.t original A
    # A'[r][c] = A[rowIndex[r][c]]
    rowIndex = dict()
    for r in range(AS):
        rowIndex[r]=[r] 
    for r in range(AS):
        # find a row k >= r s.t. A'[k][r] is not 0
        k = r
        while A[rowIndex[k]][r] == Fraction(0):
            k = k + 1
            if k==AS:
                raise Exception("Matrix is not full rank.")
        # In A', swap rows r and k
        rowIndex[k], rowIndex[r] = rowIndex[r], rowIndex[k] 
            
        for rp in range(AS):
            if r != rp:
                # replace row rp by rp-(rp[r]/r[r])r
                f: Fraction = A[rowIndex[rp]][r] / A[rowIndex[r]][r]
                b[rowIndex[rp]] = b[rowIndex[rp]] - f*b[rowIndex[r]]
                for c in range(AS):
                    A[rowIndex[rp]][c] = A[rowIndex[rp]][c] - f*A[rowIndex[r]][c]

    # Equation is reduced to Ix=b
    return [b[rowIndex[r]] for r in range(AS)]