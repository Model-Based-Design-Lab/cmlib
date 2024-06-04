'''Max-Plus Linear algebra library.'''

import re
from fractions import Fraction
from functools import reduce
from typing import (AbstractSet, Any, Dict, List, Literal, Optional, Tuple,
                    Union)

import pygraph.algorithms.accessibility as pyga
import pygraph.classes.digraph as pyg
from dataflow.maxplus.algebra import (MP_MINUSINFINITY, MP_MINUSINFINITY_STR,
                                      MPAlgebraException, mp_comp_larger,
                                      mp_op_max, mp_op_minus, mp_op_plus)
from dataflow.maxplus.cyclemean import maximum_cycle_mean
from dataflow.maxplus.starclosure import star_closure
from dataflow.maxplus.types import (TMPMatrix, TMPVector, TMPVectorList,
                                    TTimeStamp, TTimeStampList)
from dataflow.utils.visualization import weighted_graph_to_graph_viz

TThroughputValue = Union[Fraction,Literal['infinite']]

class MPException(Exception):
    '''Exceptions in the max-plus package.'''

def mp_matrix_minus_scalar(matrix: TMPMatrix, c: TTimeStamp) -> TMPMatrix:
    '''Subtract scalar from matrix element-wise.'''
    if c == MP_MINUSINFINITY:
        raise MPAlgebraException('Cannot subtract minus infinity')
    return [ [mp_op_minus(e, c) for e in r] for r in matrix]

def mp_transpose_matrix(matrix: TMPMatrix)->TMPMatrix:
    '''Transpose the matrix A.'''
    return list(map(list, zip(*matrix)))

def mp_zero_vector(n: int) -> TMPVector:
    '''Return a zero-vector (having value 0 everywhere) of size n'''
    return [Fraction(0)] * n

def mp_minus_inf_vector(n: int) -> TMPVector:
    '''Return a minus-infinity-vector (having value -inf everywhere) of size n'''
    return [MP_MINUSINFINITY] * n

def mp_inner_product(v: TMPVector, w: TMPVector)->TTimeStamp:
    '''Compute the inner product of vectors v and w.'''
    res = MP_MINUSINFINITY
    for k, val in enumerate(v):
        res = mp_op_max(res, mp_op_plus(val, w[k]))
    return res

def mp_multiply_matrices(ma: TMPMatrix, mb: TMPMatrix)->TMPMatrix:
    '''Multiply matrices A and B. Assumes they are of compatible sizes without checking.'''
    mbt = mp_transpose_matrix(mb)
    return [[mp_inner_product(ra, rb) for rb in mbt] for ra in ma]

def mp_multiply_matrix_vector_sequence(ma: TMPMatrix, x: TMPVectorList) -> TMPVectorList:
    '''Multiply every vector in x with the matrix A and return the results as a list of vectors.'''
    return [mp_multiply_matrix_vector(ma,v) for v in x]

def mp_multiply_matrix_vector(ma: TMPMatrix, x: TMPVector) -> TMPVector:
    '''Multiply vector x with the matrix A and return the result.'''
    return [ mp_inner_product(ra, x) for ra in ma]

def mp_max_vectors(x: TMPVector, y: TMPVector) -> TMPVector:
    '''Compute the maximum of two vectors. If the vectors are of different sizes,
    the shorter vector is implicitly extended with -inf.'''
    if len(x) > len(y):
        y = y + [MP_MINUSINFINITY] * (len(x) - len(y))
    if len(y) > len(x):
        x = x + [MP_MINUSINFINITY] * (len(y) - len(x))
    return [mp_op_max(x[k], y[k]) for k in range(len(x))]

def mp_add_vectors(x: TMPVector, y: TMPVector) -> TMPVector:
    '''Compute the sum of two vectors. If the vectors are of different sizes,
    the shorter vector is implicitly extended with -inf.'''
    if len(x) > len(y):
        y = y + [MP_MINUSINFINITY] * (len(x) - len(y))
    if len(y) > len(x):
        x = x + [MP_MINUSINFINITY] * (len(y) - len(x))
    return [mp_op_plus(x[k], y[k]) for k in range(len(x))]

def mp_scale_vector(c: TTimeStamp, x: TMPVector) -> TMPVector:
    '''Scale the vector x by scalar c.'''
    return [mp_op_plus(c, x[k]) for k in range(len(x))]

def mp_stack_vectors(x: TMPVector, y: TMPVector)->TMPVector:
    '''Stack two vectors x and y to the vector [x' y']'.'''
    return x + y

def mp_max_matrices(ma: TMPMatrix, mb: TMPMatrix) -> TMPMatrix:
    '''Compute the maximum of two matrices. Assumes without checking that the matrices
    are of equal size.'''
    res = []
    rows = len(ma)
    if rows == 0:
        cols = 0
    else:
        cols = len(ma[0])
    for r in range(rows):
        r_res = []
        for c in range(cols):
            r_res.append(mp_op_max(ma[r][c], mb[r][c]))
        res.append(r_res)
    return res

def mp_parse_number(e: str, mi_str: str = MP_MINUSINFINITY_STR) -> TTimeStamp:
    '''Parse string e as a max-plus value.'''
    if e.strip() == mi_str:
        return MP_MINUSINFINITY
    return Fraction(e).limit_denominator()

def mp_parse_vector(v:str, mi_str: str = MP_MINUSINFINITY_STR)->TMPVector:
    '''Parse string v as a max-plus vector.'''
    lst = re.sub(r"[\[\]]", "", v)
    return [mp_parse_number(e.strip(), mi_str) for e in lst.split(',')]


def mp_parse_traces(tt: str, mi_str: str = '-inf')->List[TTimeStampList]:
    '''Parse string tt as a sequence of event sequences (traces): syntax
    example: [-inf,-inf,0,-inf];[]. Returns the list of sequences.'''
    traces = tt.split(';')
    res: List[TTimeStampList] = []
    for t in traces:
        res.append(mp_parse_vector(t, mi_str))
    return res

def mp_number_of_rows(matrix: TMPMatrix) -> int:
    '''Return the number of rows of M.'''
    return len(matrix)

def mp_number_of_columns(matrix: TMPMatrix) -> int:
    '''Return the number of columns of M.'''
    if len(matrix) == 0:
        return 0
    return len(matrix[0])

def mp_matrix_to_precedence_graph(matrix: TMPMatrix, labels: Optional[List[str]] = None)-> \
    pyg.digraph:
    '''Convert a square matrix M to precedence graph. Optionally specify labels for the vertices.'''

    n = len(matrix)
    gr = pyg.digraph()
    _require_matrix_square(matrix)

    make_node = (lambda i: labels[i]) if not labels is None else (lambda i: f'n{i}')
    gr.add_nodes(labels if not labels is None else [ f'n{k}' for k in range(n)])

    def make_edge(i, j):
        return (make_node(i), make_node(j))

    for i in range(n):
        for j in range(n):
            if matrix[i][j] is not None:
                gr.add_edge(make_edge(j, i), matrix[i][j])  # type: ignore (edge weights are numbers, not int) pylint: disable=arguments-out-of-order
    return gr


def _subgraph(gr: pyg.digraph, nodes: AbstractSet[Any] ) -> pyg.digraph:
    '''Create subgraph from the set of node'''
    res = pyg.digraph()
    res.add_nodes(nodes)
    edges = [e for e in gr.edges() if (e[0] in nodes and e[1] in nodes)]
    for e in edges:
        res.add_edge(e, gr.edge_weight(e))
    return res

def mp_eigen_value(matrix: TMPMatrix) -> Union[None,Fraction]:
    '''Determine the largest eigenvalue of the matrix.'''

    # convert to precedence graph
    gr = mp_matrix_to_precedence_graph(matrix)

    # get the strongly connected components
    sccs = pyga.mutual_accessibility(gr)
    cycle_means: List[Fraction] = []
    subgraphs: List[pyg.digraph] = []
    mu = Fraction(0.0)
    for sn in ({frozenset(v) for v in sccs.values()}):
        grs = _subgraph(gr, sn)
        if len(grs.edges()) > 0:
            mu: Fraction
            mu, _, _ = maximum_cycle_mean(grs)  # type: ignore as returned cycle mean cannot be None
            subgraphs.append(grs)
        cycle_means.append(mu)

    if len(cycle_means) == 0:
        return None
    return max(cycle_means)

def lambda_to_throughput(lmb: Union[Fraction,None]) -> TThroughputValue:
    '''Convert lambda to a throughput value.'''
    if lmb is None:
        return "infinite"
    if not lmb > Fraction(0.0):
        return "infinite"
    return Fraction(1.0) / lmb


def mp_throughput(matrix: TMPMatrix) -> Union[Fraction,Literal["infinite"]]:
    '''Return the maximal throughput of a system with state matrix M.'''
    v_lambda = mp_eigen_value(matrix)
    return lambda_to_throughput(v_lambda)

def mp_generalized_throughput(matrix: TMPMatrix) -> List[TThroughputValue]:
    '''Return the maximal throughput of a system with state matrix M for each
    of the state vector element separately .'''
    size = len(matrix)
    evs, g_evs = mp_eigen_vectors(matrix)
    v_lambdas: List[TTimeStamp] = [MP_MINUSINFINITY for _ in range(size)]
    for ev in evs:
        lmb = ev[1]
        v = ev[0]
        for k in range(size):
            if not v[k]==MP_MINUSINFINITY:
                if mp_comp_larger(lmb, v_lambdas[k]):
                    v_lambdas[k] = lmb
    for gev in g_evs:
        lambdas = gev[1]
        v = gev[0]
        for k in range(size):
            if not v[k]==MP_MINUSINFINITY:
                if mp_comp_larger(lambdas[k], v_lambdas[k]):
                    v_lambdas[k] = lambdas[k]

    return [lambda_to_throughput(l) for l in v_lambdas]


def _normalized_longest_paths(gr: pyg.digraph, rootnode: Any, cycle_means_map: \
                Dict[Any,TTimeStamp]) -> \
                Tuple[Dict[Any,TTimeStamp],Dict[Any,TTimeStamp]]:

    def _normalize_dict(d: Dict[Any,TTimeStamp]):
        '''
        Normalize a dictionary with numbers or None as the co-domain, by subtracting
        the largest value from all values, so that the largest value becomes 0
        '''

        # find the largest value
        max_val: TTimeStamp = None
        for n in d:
            if d[n] is not None:
                if max_val is None:
                    max_val = d[n]
                else:
                    if d[n] > max_val: # type: ignore we know that d[n] is a Fraction
                        max_val = d[n]
        # if all values were None, nothing to be done.
        if max_val is None:
            return d

        # subtract maxVal from all values to make the result
        res = {}
        for n in d:
            if d[n] is None:
                res[n] = None
            else:
                res[n] = d[n] - max_val # type: ignore we know that d[n] is a Fraction

        return res

    # compute transitive cycle means starting from the root node only
    # (not from all original cycle means)
    # for nodes downstream from the root node also take max with their
    # local cycle mean
    tr_cycle_means_map: Dict[Any,TTimeStamp] = {n: None for n in gr.nodes()}
    tr_cycle_means_map[rootnode] = cycle_means_map[rootnode]

    # fixed-point computation
    change = True
    while change:
        change = False
        # check for all edges if the cycle mean needs updating
        for e in gr.edges():
            if tr_cycle_means_map[e[0]] is not None:
                if tr_cycle_means_map[e[1]] is None:
                    tr_cycle_means_map[e[1]] = tr_cycle_means_map[e[0]]
                    change = True
                    if cycle_means_map[e[1]] is not None:
                        if cycle_means_map[e[1]] >\
                        tr_cycle_means_map[e[1]]: # type: ignore we know that both must be Fractions
                            tr_cycle_means_map[e[1]] = cycle_means_map[e[1]]
                            change = True
                else:
                    if tr_cycle_means_map[e[0]] > tr_cycle_means_map[e[1]]:  # type: ignore
                        tr_cycle_means_map[e[1]] = tr_cycle_means_map[e[0]]
                        change = True


    length: Dict[Any, TTimeStamp] = {}
    for n in gr.nodes():
        length[n] = None
    length[rootnode] = Fraction(0.0)

    # compute the normalized longest paths, i.e., the path lengths normalized
    # by the local transitive cycle means of the sending node
    change = True
    while change:
        change = False
        for e in gr.edges():
            if length[e[0]] is not None:
                new_length = length[e[0]] + gr.edge_weight(e) - tr_cycle_means_map[e[0]]
                if length[e[1]] is None:
                    length[e[1]] = new_length
                    change = True
                else:
                    if length[e[1]] < new_length:
                        length[e[1]] = new_length
                        change = True

    # return the path lengths and the transitive cycle means
    return _normalize_dict(length), tr_cycle_means_map



def mp_eigen_vectors(matrix: TMPMatrix) -> Tuple[List[Tuple[TMPVector,Fraction]], \
                                                 List[Tuple[TMPVector,TMPVector]]]:
    '''
    Compute the eigenvectors of a square matrix.
    Return a pair of a list of eigenvector and a list of generalized eigenvectors.
    '''
    # compute the precedence graphs
    # compute the strongly connected components and their cycle means
    # for each scc, determine a (generalized) eigenvector

    def _is_regular_eigen_value(evv: Dict[Any,TTimeStamp])->bool:
        '''Check if the set of nodes have at most one eigenvalue'''
        value = None
        for n in evv:
            if evv[n] is not None:
                evv_n: Fraction = evv[n]  # type: ignore
                if value is None:
                    value = evv[n]
                else:
                    if not value==evv_n:
                        return False
        return True

    def _as_eigen_value(evv: Dict[Any,TTimeStamp])->Fraction:
        for n in evv:
            if evv[n] is not None:
                evv_n: Fraction = evv[n]  # type: ignore
                return evv_n
        raise MPException("Eigenvalue cannot be minus infinity.")

    def _as_generalized_eigen_value(evv: Dict[Any,TTimeStamp])->TTimeStampList:
        evl: TTimeStampList = []
        for n in gr.nodes():
            evl.append(evv[n])
        return evl

    gr = mp_matrix_to_precedence_graph(matrix)

    # compute the strongly connected components of the precedence graph
    sccs = pyga.mutual_accessibility(gr)

    # keep lists of cycleMeans and subgraphs of the SCCs
    cycle_means: List[TTimeStamp] = []
    subgraphs: List[pyg.digraph] = []

    # count the SCCs with k
    k: int = 0

    # a map from nodes of the precedence graph to their SCC index
    scc_map: Dict[Any,int] = {}
    # map from SCC index to a node of the SCC
    scc_map_inv: Dict[int,Any] = {}
    # a list such that item k is a critical node of SCC k
    critical_nodes: List[Any] = list()
    # a map from nodes to the cycle mean of their SCC
    cycle_means_map: Dict[Any,TTimeStamp] = {}

    # for each of the sets of nodes of the SCCs
    for sn in ({frozenset(v) for v in sccs.values()}):
        scc_map_inv[k] = next(iter(sn))

        # extract subgraph of the SCC
        grs = _subgraph(gr, sn)
        subgraphs.append(grs)

        if len(grs.edges()) > 0:
            # if the subgraph has edges, it has cycles...
            # compute the MCM of the subgraph and one critical node on the cycle
            mu, _, critical_node = maximum_cycle_mean(grs)
            critical_nodes.append(critical_node)
            cycle_means.append(mu)
        else:
            # if the SCC has no edges, its cycle mean is None = '-inf'
            # add arbitrary "critical" node
            critical_nodes.append(grs.nodes()[0])
            cycle_means.append(None)

        # do the administration for each of the nodes
        for n in sn:
            scc_map[n] = k
            cycle_means_map[n] = cycle_means[k]
        k += 1

    # trCycleMeans keeps the transitive cycle means, i.e., the maximum of the own SCC
    # cycle mean, of the cycle mean of an upstream SCC, an SCC from which the current
    # SCC can be reached.
    tr_cycle_means = cycle_means.copy()
    change = True
    while change:
        change = False
        for e in gr.edges():
            # check cycle means or None if scc has no cycle
            if tr_cycle_means[scc_map[e[1]]] is None or mp_comp_larger(tr_cycle_means[\
                scc_map[e[0]]], tr_cycle_means[scc_map[e[1]]]):
                change = True
                tr_cycle_means[scc_map[e[1]]] = tr_cycle_means[scc_map[e[0]]]

    # two lists to keep the results
    eigen_vectors: List[Tuple[TMPVector,Fraction]] = []
    gen_eigen_vectors: List[Tuple[TMPVector,TMPVector]] = []

    # for each of the SCC subgraphs that have a cycle mean that is not -inf
    for k in range(len(subgraphs)):
        if cycle_means[k] is not None:
            # compute eigenvector and generalized eigenvalue from the SCC as the longest paths
            # in the normalized graph from the criticalNode of the SCC
            ev, evv = _normalized_longest_paths(gr, critical_nodes[k], cycle_means_map)
            # collect the eigenvector in the list evl
            evl: TMPVector = []
            # for each of the nodes of the precedence graph
            for n in gr.nodes():
                # append the path length to node n
                evl.append(ev[n])
            # check if the result is a normal eigenvalue or only a generalized eigenvalue
            # and process the results accordingly
            if _is_regular_eigen_value(evv):
                eigen_vectors.append((evl, _as_eigen_value(evv)))
            else:
                gen_eigen_vectors.append((evl, _as_generalized_eigen_value(evv)))

    # return the results
    return (eigen_vectors, gen_eigen_vectors)

def mp_precedence_graph(matrix: TMPMatrix, labels: List[str])->pyg.digraph:
    '''Determine the precedence graph of the matrix M using the labels for vertices.'''
    return mp_matrix_to_precedence_graph(matrix, labels)

def mp_precedence_graph_graphviz(matrix: TMPMatrix, labels: List[str])->str:
    '''Determine the precedence graph as a Graphviz string of the matrix M using
    the labels for vertices.'''
    gr = mp_matrix_to_precedence_graph(matrix, labels)
    return weighted_graph_to_graph_viz(gr)

def mp_star_closure(matrix: TMPMatrix) -> TMPMatrix:
    '''Determine the star close of the matrix M. A PositiveCycleException is raised
    if the closure does not exist.'''
    return star_closure(matrix)

def mp_convolution(s: TTimeStampList, t: TTimeStampList)->TTimeStampList:
    '''Compute the convolution of the event sequences s and t'''
    res: TTimeStampList = []
    l: int = min(len(s), len(t))
    for k in range(l):
        v = reduce(lambda mx, n: mp_op_max(mx, mp_op_plus(s[n], t[k-n])), range(k+1), \
                   MP_MINUSINFINITY) # pylint: disable=cell-var-from-loop
        res.append(v)
    return res

def mp_max_event_sequences(es1: TTimeStampList, es2: TTimeStampList)->TTimeStampList:
    '''Determine the maximum of two event sequences.'''
    return mp_max_vectors(es1, es2)


def mp_max_vector_sequences(vs1: TMPVectorList, vs2: TMPVectorList) -> TMPVectorList:
    '''Determine the maximum of two vector sequences.'''
    return [mp_max_vectors(vs1[k], vs2[k]) for k in range(min(len(vs1), len(vs2)))]

def mp_split_sequence(seq: TTimeStampList, n: int)->List[TTimeStampList]:
    '''Split event sequence seq into n interleaving sub sequences. E.g., with
    seq=[2, 3, 6, 9, 11, 12] and n=3, the result is [[2,9],[3,11],[6,12]]'''
    return [seq[k::n] for k in range(n)]

def mp_merge_sequences(seqs: List[TTimeStampList])->TTimeStampList:
    '''Merge sequences.'''
    return [item for sublist in zip(*seqs) for item in sublist]

def mp_delay(seq: TTimeStampList, n: int) -> TTimeStampList:
    '''Delay an event sequence by n samples.'''
    return ([MP_MINUSINFINITY] * n) + seq

def mp_scale(seq: TTimeStampList, c: TTimeStamp)->TTimeStampList:
    '''Scale an event sequence by a scalar c'''
    return [mp_op_plus(v, c) for v in seq]

def _require_matrix_square(matrix: TMPMatrix):
    '''Raise an exception of the matrix M is not square.'''
    if mp_number_of_columns(matrix) != mp_number_of_rows(matrix):
        raise MPException("Matrix should be square.")
