'''Compute max-plus star closure.'''

from fractions import Fraction

from dataflow.maxplus.algebra import (MP_MINUSINFINITY, mp_comp_larger,
                                      mp_op_max, mp_op_plus)
from dataflow.maxplus.types import TMPMatrix


class PositiveCycleException(Exception):
    '''Exception when the closure does not exist due to a positive cycle,.'''

def star_closure(matrix: TMPMatrix)->TMPMatrix:
    '''Compute the star closure of the matrix.'''

    n = len(matrix)

    # copy the matrix
    res = [r.copy() for r in matrix]

    # // k - intermediate node
    for k in range(n):
        for u in range(n):
            for v in range(n):

                if u == v:
                    extra = Fraction(0.0)
                else:
                    extra = MP_MINUSINFINITY
                path_u2v = mp_op_max(res[v][u], extra)
                path_u2k = res[k][u]
                path_k2v = res[v][k]

                path_u2v_candidate = mp_op_plus(path_u2k, path_k2v)
                if mp_comp_larger(path_u2v_candidate, path_u2v):
                    path_u2v = path_u2v_candidate
                res[v][u] = path_u2v

    for k in range(n):
        if mp_comp_larger(res[k][k], Fraction(0.0)):
            raise PositiveCycleException("Star Closure does not exist")

    return res
