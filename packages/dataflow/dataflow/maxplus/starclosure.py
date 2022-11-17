from dataflow.maxplus.algebra import MP_MAX, MP_PLUS, MP_LARGER, MP_MINUSINFINITY, NumericalEpsilon
from dataflow.maxplus.types import TMPMatrix

class PositiveCycleException(Exception):
    pass

def starClosure(M: TMPMatrix)->TMPMatrix:

    N = len(M)

    # copy the matrix
    res = [r.copy() for r in M]

    # // k - intermediate node
    for k in range(N):
        for u in range(N):
            for v in range(N):

                if u == v:
                    extra = 0.0
                else:
                    extra = MP_MINUSINFINITY
                path_u2v = MP_MAX(res[v][u], extra)
                path_u2k = res[k][u]
                path_k2v = res[v][k]

                path_u2v_candidate = MP_PLUS(path_u2k, path_k2v)
                if MP_LARGER(path_u2v_candidate, path_u2v):
                    path_u2v = path_u2v_candidate
                res[v][u] = path_u2v

    for k in range(N):
        if MP_LARGER(res[k][k], 0.0):
            raise PositiveCycleException("Star Closure does not exist")

    return res
