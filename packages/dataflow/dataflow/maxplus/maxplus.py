from typing import AbstractSet, Any, Dict, List, Literal, Optional, Tuple, Union
from dataflow.utils.visualization import weightedGraphToGraphViz
import pygraph.classes.digraph  as pyg
import pygraph.algorithms.accessibility as pyga
import re
from functools import reduce
from dataflow.maxplus.cyclemean import maximumCycleMean
from dataflow.maxplus.starclosure import starClosure
from fractions import Fraction
from  dataflow.maxplus.algebra import MP_MAX, MP_PLUS, MP_MINUS, MP_LARGER, MP_MINUSINFINITY, MPAlgebraException
from dataflow.maxplus.types import TMPMatrix, TMPVector, TTimeStamp, TMPVectorList, TTimeStampList

class MPException(Exception):
    pass

def mpMatrixMinusScalar(M: TMPMatrix, c) -> TMPMatrix:
    '''Subtract scalar from matrix element-wise.'''
    if c == MP_MINUSINFINITY:
        raise MPAlgebraException('Cannot subtract minus infinity')
    return [ [MP_MINUS(e, c) for e in r] for r in M]

def mpTransposeMatrix(A: TMPMatrix)->TMPMatrix:
    '''Transpose the matrix A.'''
    return list(map(list, zip(*A)))

def mpZeroVector(n: int) -> TMPVector:
    '''Return a zero-vector (having value 0 everywhere) of size n'''
    return [Fraction(0)] * n

def mpMinusInfVector(n: int) -> TMPVector:
    '''Return a minus-infinity-vector (having value -inf everywhere) of size n'''
    return [MP_MINUSINFINITY] * n

def mpInnerProduct(v: TMPVector, w: TMPVector)->TTimeStamp:
    '''Compute the inner product of vectors v and w.'''
    res = MP_MINUSINFINITY
    for k in range(len(v)):
        res = MP_MAX(res, MP_PLUS(v[k], w[k]))
    return res

def mpMultiplyMatrices(A: TMPMatrix, B: TMPMatrix)->TMPMatrix:
    '''Multiply matrices A and B. Assumes they are of compatible sizes without checking.'''
    BT = mpTransposeMatrix(B)
    return [[mpInnerProduct(ra, rb) for rb in BT] for ra in A]

def mpMultiplyMatrixVectorSequence(A: TMPMatrix, x: TMPVectorList) -> TMPVectorList:
    '''Multiply every vector in x with the matrix A and return the results as a list of vectors.'''
    return [mpMultiplyMatrixVector(A,v) for v in x]

def mpMultiplyMatrixVector(A: TMPMatrix, x: TMPVector) -> TMPVector:
    '''Multiply vector x with the matrix A and return the result.'''
    return [ mpInnerProduct(ra, x) for ra in A]

def mpMaxVectors(x: TMPVector, y: TMPVector) -> TMPVector:
    '''Compute the maximum of two vectors. If the vectors are of different sizes, the shorter vector is implicitly extended with -inf.'''
    if len(x) > len(y):
        y = y + [MP_MINUSINFINITY] * (len(x) - len(y))
    if len(y) > len(x):
        x = x + [MP_MINUSINFINITY] * (len(y) - len(x))
    return [MP_MAX(x[k], y[k]) for k in range(len(x))]

def mpAddVectors(x: TMPVector, y: TMPVector) -> TMPVector:
    '''Compute the sum of two vectors. If the vectors are of different sizes, the shorter vector is implicitly extended with -inf.'''
    if len(x) > len(y):
        y = y + [MP_MINUSINFINITY] * (len(x) - len(y))
    if len(y) > len(x):
        x = x + [MP_MINUSINFINITY] * (len(y) - len(x))
    return [MP_PLUS(x[k], y[k]) for k in range(len(x))]

def mpScaleVector(c: TTimeStamp, x: TMPVector) -> TMPVector:
    '''Scale the vector x by scalar c.'''
    return [MP_PLUS(c, x[k]) for k in range(len(x))]

def mpStackVectors(x: TMPVector, y: TMPVector)->TMPVector:
    '''Stack two vectors x and y to the vector [x' y']'.'''
    return x + y

def mpMaxMatrices(A: TMPMatrix, B: TMPMatrix) -> TMPMatrix:
    '''Compute the maximum of two matrices. Assumes without checking that the matrices are of equal size.'''
    res = []
    rows = len(A)
    if rows == 0:
        cols = 0
    else:
        cols = len(A[0])
    for r in range(rows):
        rRes = []
        for c in range(cols):
            rRes.append(MP_MAX(A[r][c], B[r][c]))
        res.append(rRes)
    return res

def mpElementToString(x: TTimeStamp, miStr: str = '-inf')->str:
    '''Return a 6-character wide string representation of the max-plus element x, using miStr, defaulting to '-inf' to represent minus infinity.'''
    # TODO: make the width and formatting more flexible. Not even sure it is 6, based on miStr
    if x is MP_MINUSINFINITY:
        return '  '+miStr
    return '{}'.format(x)

def mpVectorToString(v: TMPVector)->str:
    '''Return string representation of the vector v.'''
    return '[ {} ]'.format(' '.join([mpElementToString(x) for x in v]))

def mpParseNumber(e: str, miStr: str = '-inf') -> TTimeStamp:
    '''Parse string e as a max-plus value.'''
    if e.strip() == miStr:
        return MP_MINUSINFINITY
    return Fraction(e).limit_denominator()

def mpParseVector(v:str, miStr: str = '-inf')->TMPVector:
    '''Parse string v as a max-plus vector.'''
    lst = re.sub(r"[\[\]]", "", v)
    return [mpParseNumber(e.strip(), miStr) for e in lst.split(',')]


def mpParseTraces(tt: str, miStr: str = '-inf')->List[TTimeStampList]:
    '''Parse string tt as a sequence of event sequences (traces): syntax example: [-inf,-inf,0,-inf];[]. Returns the list of sequences.'''
    traces = tt.split(';')
    res: List[TTimeStampList] = []
    for t in traces:
        res.append(mpParseVector(t, miStr))
    return res

def mpNumberOfRows(M: TMPMatrix) -> int:
    '''Return the number of rows of M.'''
    return len(M)

def mpNumberOfColumns(M: TMPMatrix) -> int:
    '''Return the number of columns of M.'''
    if len(M) == 0:
        return 0
    return len(M[0])

def printMPMatrix(M: TMPMatrix):
    '''Print matrix M to the console.'''
    print('[', end="")
    print('\n'.join([mpVectorToString(row) for row in M]), end="")
    print(']')

def printMPVectorList(vl: TMPVectorList):
    '''Print list of vectors to the console.'''
    print('[', end="")
    print('\n'.join(mpVectorToString(v) for v in vl), end="")
    print(']')

def mpMatrixToPrecedenceGraph(M: TMPMatrix, labels: Optional[List[str]] = None)->pyg.digraph:
    '''Convert a square matrix M to precedence graph. Optionally specify labels for the vertices.'''

    N = len(M)
    gr = pyg.digraph()
    _requireMatrixSquare(M)
    
    make_node = (lambda i: labels[i]) if not labels is None else (lambda i: 'n{}'.format(i))
    gr.add_nodes(labels if not labels is None else [ 'n{}'.format(k) for k in range(N)])
    
    make_edge = lambda i,j: (make_node(i), make_node(j))
    for i in range(N):
        for j in range(N):
            if M[i][j] is not None:
                # print(j,i,M[i][j])
                gr.add_edge(make_edge(j, i), M[i][j])  # type: ignore (edge weights are numbers, not int)
    return gr


def _subgraph(gr: pyg.digraph, nodes: AbstractSet[Any] ) -> pyg.digraph:
    '''Create subgraph from the set of node'''
    res = pyg.digraph()
    res.add_nodes(nodes)
    E = [e for e in gr.edges() if (e[0] in nodes and e[1] in nodes)]
    for e in E:
        res.add_edge(e, gr.edge_weight(e))
    return res

def mpEigenValue(M: TMPMatrix) -> Union[None,Fraction]:
    '''Determine the largest eigenvalue of the matrix.'''
    
    # convert to precedence graph
    gr = mpMatrixToPrecedenceGraph(M)

    # get the strongly connected components
    sccs = pyga.mutual_accessibility(gr)
    cycleMeans: List[Fraction] = []
    subgraphs: List[pyg.digraph] = []
    mu = Fraction(0.0)
    for sn in ({frozenset(v) for v in sccs.values()}):
        grs = _subgraph(gr, sn)
        if len(grs.edges()) > 0:
            mu: Fraction
            mu, _, _ = maximumCycleMean(grs)  # type: ignore as returned cycle mean cannot be None
            subgraphs.append(grs)
        cycleMeans.append(mu)

    if len(cycleMeans) == 0:
        return None
    return max(cycleMeans)

def mpThroughput(M: TMPMatrix) -> Union[Fraction,Literal["infinite"]]:
    '''Return the maximal throughput of a system with state matrix M.'''
    vLambda = mpEigenValue(M)
    if vLambda is None:
        return "infinite"
    if not vLambda> Fraction(0.0):
        return "infinite"
    return Fraction(1.0) / vLambda

def _normalizedLongestPaths(gr: pyg.digraph, rootnode: Any, cycleMeansMap: Dict[Any,TTimeStamp]) -> Tuple[Dict[Any,TTimeStamp],Dict[Any,TTimeStamp]]:

    def _normalizeDict(d: Dict[Any,TTimeStamp]):
        '''
        Normalize a dictionary with numbers or None as the co-domain, by subtracting the largest value from all values, so that the largest value becomes 0
        '''

        # find the largest value
        maxVal: TTimeStamp = None
        for n in d:
            if d[n] is not None:
                if maxVal is None:
                    maxVal = d[n]
                else:
                    if d[n] > maxVal: # type: ignore we know that d[n] is a float
                        maxVal = d[n]
        # if all values were None, nothing to be done.
        if maxVal is None:
            return d
        
        # subtract maxVal from all values to make the result
        res = dict()
        for n in d:
            if d[n] is None:
                res[n] = None
            else:
                res[n] = d[n] - maxVal # type: ignore we know that d[n] is a float

        return res

    # compute transitive cycle means starting from the root node only
    # (not from all original cycle means)
    # for nodes downstream from the root node also take max with their
    # local cycle mean
    trCycleMeansMap: Dict[Any,TTimeStamp] = dict([(n, None) for n in gr.nodes()])
    trCycleMeansMap[rootnode] = cycleMeansMap[rootnode]
    
    # fixed-point computation
    change = True
    while change:
        change = False
        # check for all edges if the cycle mean needs updating
        for e in gr.edges():
            if trCycleMeansMap[e[0]] is not None:
                if trCycleMeansMap[e[1]] is None:
                    trCycleMeansMap[e[1]] = trCycleMeansMap[e[0]]
                    change = True
                    if cycleMeansMap[e[1]] is not None:
                        if cycleMeansMap[e[1]] > trCycleMeansMap[e[1]]: # type: ignore we know that both must be floats
                            trCycleMeansMap[e[1]] = cycleMeansMap[e[1]]
                            change = True
                else:
                    if trCycleMeansMap[e[0]] > trCycleMeansMap[e[1]]:  # type: ignore
                        trCycleMeansMap[e[1]] = trCycleMeansMap[e[0]]
                        change = True


    length: Dict[Any, TTimeStamp] = dict()
    for n in gr.nodes():
        length[n] = None
    length[rootnode] = Fraction(0.0)

    # compute the normalized longest paths, i.e., the path lengths normalized by the local transitive cycle means of the sending node
    change = True
    while change:
        change = False
        for e in gr.edges():
            if length[e[0]] is not None:
                newLength = length[e[0]] + gr.edge_weight(e) - trCycleMeansMap[e[0]]
                if length[e[1]] is None:
                    length[e[1]] = newLength
                    change = True
                else:
                    if length[e[1]] < newLength:
                        length[e[1]] = newLength
                        change = True

    # return the path lengths and the transitive cycle means
    return _normalizeDict(length), trCycleMeansMap



def mpEigenVectors(M: TMPMatrix) -> Tuple[List[Tuple[TMPVector,float]],List[Tuple[TMPVector,TMPVector]]]:
    '''
    Compute the eigenvectors of a square matrix.
    Return a pair of a list of eigenvector and a list of generalized eigenvectors.
    '''
    # compute the precedence graphs
    # compute the strongly connected components and their cycle means
    # for each scc, determine a (generalized) eigenvector

    def _isRegularEigenValue(evv: Dict[Any,TTimeStamp])->bool:
        '''Check if the set of nodes have at most one eigenvalue'''
        value = None
        for n in evv:
            if evv[n] is not None:
                evv_n: float = evv[n]  # type: ignore
                if value is None:
                    value = evv[n]
                else:
                    if not value==evv_n:
                        return False
        return True

    def _asEigenValue(evv: Dict[Any,TTimeStamp])->float:
        for n in evv:
            if evv[n] is not None:
                evv_n: float = evv[n]  # type: ignore
                return evv_n
        raise MPException("Eigenvalue cannot be minus infinity.")

    def _asGeneralizedEigenValue(evv: Dict[Any,TTimeStamp])->TTimeStampList:
        evl: TTimeStampList = []
        for n in gr.nodes():
            evl.append(evv[n])
        return evl

    gr = mpMatrixToPrecedenceGraph(M)

    # compute the strongly connected components of the precedence graph
    sccs = pyga.mutual_accessibility(gr)
    
    # keep lists of cycleMeans and subgraphs of the SCCs
    cycleMeans: List[TTimeStamp] = []
    subgraphs: List[pyg.digraph] = []
    
    # count the SCCs with k
    k: int = 0
    
    # a map from nodes of the precedence graph to their SCC index
    sccMap: Dict[Any,int] = dict()
    # map from SCC index to a node of the SCC
    sccMapInv: Dict[int,Any] = dict()
    # a list such that item k is a critical node of SCC k
    criticalNodes: List[Any] = list()
    # a map from nodes to the cycle mean of their SCC
    cycleMeansMap: Dict[Any,TTimeStamp] = dict()

    # for each of the sets of nodes of the SCCs
    for sn in ({frozenset(v) for v in sccs.values()}):
        sccMapInv[k] = next(iter(sn))

        # extract subgraph of the SCC
        grs = _subgraph(gr, sn)
        subgraphs.append(grs)
        
        if len(grs.edges()) > 0:
            # if the subgraph has edges, it has cycles...
            # compute the MCM of the subgraph and one critical node on the cycle
            mu, _, criticalNode = maximumCycleMean(grs)
            criticalNodes.append(criticalNode)
            cycleMeans.append(mu)
        else:
            # if the SCC has no edges, its cycle mean is None = '-inf'
            # add arbitrary "critical" node
            criticalNodes.append(grs.nodes()[0])
            cycleMeans.append(None)
        
        # do the administration for each of the nodes
        for n in sn:
            sccMap[n] = k
            cycleMeansMap[n] = cycleMeans[k]
        k += 1

    # trCycleMeans keeps the transitive cycle means, i.e., the maximum of the own SCC cycle mean, of the cycle mean of an upstream SCC, an SCC from which the current SCC can be reached.
    trCycleMeans = cycleMeans.copy()
    change = True
    while change:
        change = False
        for e in gr.edges():
            # check cycle means or None if scc has no cycle
            if trCycleMeans[sccMap[e[1]]] is None or MP_LARGER(trCycleMeans[sccMap[e[0]]], trCycleMeans[sccMap[e[1]]]):
                change = True
                trCycleMeans[sccMap[e[1]]] = trCycleMeans[sccMap[e[0]]]

    # two lists to keep the results
    eigenVectors: List[Tuple[TMPVector,float]] = []
    genEigenVectors: List[Tuple[TMPVector,TMPVector]] = []
    
    # for each of the SCC subgraphs that have a cycle mean that is not -inf
    for k in range(len(subgraphs)):
        if cycleMeans[k] is not None:
            # compute eigenvector and generalized eigenvalue from the SCC as the longest paths 
            # in the normalized graph from the criticalNode of the SCC
            ev, evv = _normalizedLongestPaths(gr, criticalNodes[k], cycleMeansMap)
            # collect the eigenvector in the list evl
            evl: TMPVector = []
            # for each of the nodes of the precedence graph
            for n in gr.nodes():
                # append the path length to node n
                evl.append(ev[n])
            # check if the result is a normal eigenvalue or only a generalized eigenvalue
            # and process the results accordingly
            if _isRegularEigenValue(evv):
                eigenVectors.append((evl, _asEigenValue(evv)))
            else:
                genEigenVectors.append((evl, _asGeneralizedEigenValue(evv)))
    
    # return the results
    return (eigenVectors, genEigenVectors)

def mpPrecedenceGraph(M: TMPMatrix, labels: List[str])->pyg.digraph:
    '''Determine the precedence graph of the matrix M using the labels for vertices.'''
    return mpMatrixToPrecedenceGraph(M, labels)

def mpPrecedenceGraphGraphviz(M: TMPMatrix, labels: List[str])->str:
    '''Determine the precedence graph as a Graphviz string of the matrix M using the labels for vertices.'''
    gr = mpMatrixToPrecedenceGraph(M, labels)
    return weightedGraphToGraphViz(gr)

def mpStarClosure(M: TMPMatrix) -> TMPMatrix:
    '''Determine the star close of the matrix M. A PositiveCycleException is raised if the closure does not exist.'''
    return starClosure(M)

def mpConvolution(s: TTimeStampList, t: TTimeStampList)->TTimeStampList:
    '''Compute the convolution of the event sequences s and t'''
    res: TTimeStampList = []
    l: int = min(len(s), len(t))
    for k in range(l):
        v = reduce(lambda mx, n: MP_MAX(mx, MP_PLUS(s[n], t[k-n])), range(k+1), MP_MINUSINFINITY)
        res.append(v)
    return res

def mpMaxEventSequences(es1: TTimeStampList, es2: TTimeStampList)->TTimeStampList:
    '''Determine the maximum of two event sequences.'''
    return mpMaxVectors(es1, es2)


def mpMaxVectorSequences(vs1: TMPVectorList, vs2: TMPVectorList) -> TMPVectorList:
    '''Determine the maximum of two vector sequences.'''
    return [mpMaxVectors(vs1[k], vs2[k]) for k in range(min(len(vs1), len(vs2)))]

def mpSplitSequence(seq: TTimeStampList, n: int)->List[TTimeStampList]:
    '''Split event sequence seq into n interleaving sub sequences. E.g., with seq=[2, 3, 6, 9, 11, 12] and n=3, the result is [[2,9],[3,11],[6,12]]'''
    return [seq[k::n] for k in range(n)] 

def mpMergeSequences(seqs: List[TTimeStampList])->TTimeStampList:
    return [item for sublist in zip(*seqs) for item in sublist]
    
def mpDelay(seq: TTimeStampList, n: int) -> TTimeStampList:
    '''Delay an event sequence by n samples.'''
    return ([MP_MINUSINFINITY] * n) + seq

def mpScale(seq: TTimeStampList, c: TTimeStamp)->TTimeStampList:
    '''Scale an event sequence by a scalar c'''
    return [MP_PLUS(v, c) for v in seq]

def _requireMatrixSquare(M: TMPMatrix):
    '''Raise an exception of the matrix M is not square.'''
    if mpNumberOfColumns(M) != mpNumberOfRows(M):
        raise MPException("Matrix should be square.")