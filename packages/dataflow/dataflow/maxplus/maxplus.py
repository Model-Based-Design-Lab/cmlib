from typing import Any
from dataflow.utils.visualization import weightedGraphToGraphViz
import pygraph.classes.digraph  as pyg
import pygraph.algorithms.accessibility as pyga
import re
from functools import reduce
from dataflow.maxplus.cyclemean import maximumCycleMean
from dataflow.maxplus.starclosure import starClosure

from  dataflow.maxplus.algebra import MP_MAX, MP_PLUS, MP_MINUS, MP_MINUSINFINITY, MPAlgebraException, significantlySmaller

TTimeStamp = float|None
TTimeStampList = list[TTimeStamp]
TMPVector = TTimeStampList
TMPVectorList = list[TMPVector]
TMPMatrix = TMPVectorList

class MPException(Exception):
    pass

def mpMatrixMinusScalar(M, c):
    if c == MP_MINUSINFINITY:
        raise MPAlgebraException('Cannot subtract minus infinity')
    return [ [MP_MINUS(e, c) for e in r] for r in M]

def mpTransposeMatrix(A)->TMPMatrix:
    return list(map(list, zip(*A)))

def mpZeroVector(n):
    return [0] * n

def mpMinusInfVector(n):
    return [MP_MINUSINFINITY] * n

def mpInnerProduct(v, w):
    res = MP_MINUSINFINITY
    for k in range(len(v)):
        res = MP_MAX(res, MP_PLUS(v[k], w[k]))
    return res

def mpMultiplyMatrices(A, B)->TMPMatrix:
    BT = mpTransposeMatrix(B)
    return [[mpInnerProduct(ra, rb) for rb in BT] for ra in A]

def mpMultiplyMatrixVectorSequence(A, x) -> TMPMatrix:
    return [mpMultiplyMatrixVector(A,v) for v in x]

def mpMultiplyMatrixVector(A, x) -> TMPVector:
    return [ mpInnerProduct(ra, x) for ra in A]

def mpMaxVectors(x, y):
    if len(x) > len(y):
        y = y + [None] * (len(x) - len(y))
    if len(y) > len(x):
        x = x + [None] * (len(y) - len(x))
    return [MP_MAX(x[k], y[k]) for k in range(len(x))]

def mpAddVectors(x, y):
    if len(x) > len(y):
        y = y + [None] * (len(x) - len(y))
    if len(y) > len(x):
        x = x + [None] * (len(y) - len(x))
    return [MP_PLUS(x[k], y[k]) for k in range(len(x))]

def mpScaleVector(c, x):
    return [MP_PLUS(c, x[k]) for k in range(len(x))]


def mpStackVectors(x, y):
    return x + y

def mpMaxMatrices(A, B):
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

def mpElement(x):
    if x is None:
        return '  -inf'
    return '{:6.4f}'.format(x)

def mpVector(v):
    return '[ {} ]'.format(' '.join([mpElement(x) for x in v]))

def mpParseNumber(e):
    if e.strip() == '-inf':
        return MP_MINUSINFINITY
    return float(e)

def mpParseVector(v):
    lst = re.sub(r"[\[\]]", "", v)
    return [mpParseNumber(e.strip()) for e in lst.split(',')]


def mpParseTraces(tt):
    '''[-inf,-inf,0,-inf];[]'''
    traces = tt.split(';')
    res = []
    for t in traces:
        res.append(mpParseVector(t))
    return res


def mpNumberOfRows(M):
    return len(M)

def mpNumberOfColumns(M):
    if len(M) == 0:
        return 0
    return len(M[0])

def printMPMatrix(M):
    print('[', end="")
    print('\n'.join([mpVector(row) for row in M]), end="")
    print(']')

def mpMatrixToPrecedenceGraph(M, labels = None):
    N = len(M)
    gr = pyg.digraph()
    
    make_node = (lambda i: labels[i]) if not labels is None else (lambda i: 'n{}'.format(i))
    gr.add_nodes(labels if not labels is None else [ 'n{}'.format(k) for k in range(N)])
    
    make_edge = lambda i,j: (make_node(i), make_node(j))
    for i in range(N):
        for j in range(N):
            if M[i][j] is not None:
                # print(j,i,M[i][j])
                gr.add_edge(make_edge(j, i), M[i][j])
    return gr


def _subgraph(gr, nodes):
    res = pyg.digraph()
    res.add_nodes(nodes)
    E = [e for e in gr.edges() if (e[0] in nodes and e[1] in nodes)]
    for e in E:
        res.add_edge(e, gr.edge_weight(e))
    return res

def mpEigenValue(M):
    gr = mpMatrixToPrecedenceGraph(M)

    sccs = pyga.mutual_accessibility(gr)
    cycleMeans = []
    subgraphs = []
    mu = 0.0
    for sn in ({frozenset(v) for v in sccs.values()}):
        grs = _subgraph(gr, sn)
        if len(grs.edges()) > 0:
            mu, _, _ = maximumCycleMean(grs)
            subgraphs.append(grs)
        cycleMeans.append(mu)

    if len(cycleMeans) == 0:
        return 0.0
    return max(cycleMeans)

def mpThroughput(M):

    lmbda = mpEigenValue(M)
    if lmbda == 0.0:
        return "infinite"
    return 1.0 / lmbda

def _normalizedLongestPaths(gr, rootnode, cycleMeansMap) -> tuple[dict[Any,float|None],dict[Any,float|None]]:

    def _normalizeDict(d):
        '''
        Normalize a dictionary with numbers or None as the co-domain, by subtracting the largest value from all values, so that the largest value becomes 0
        '''

        # find the largest value
        maxVal = None
        for n in d:
            if d[n] is not None:
                if maxVal is None:
                    maxVal = d[n]
                else:
                    if d[n] > maxVal:
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
                res[n] = d[n] - maxVal

        return res

    # compute transitive cycle means starting from the root node only
    # (not from all original cycle means)
    # for nodes downstream from the root node also take max with their
    # local cycle mean
    trCycleMeansMap: dict[Any,float|None] = dict([(n, None) for n in gr.nodes()])
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
                        if cycleMeansMap[e[1]] > trCycleMeansMap[e[1]]:
                            trCycleMeansMap[e[1]] = cycleMeansMap[e[1]]
                            change = True
                else:
                    if trCycleMeansMap[e[0]] > trCycleMeansMap[e[1]]:  # type: ignore
                        trCycleMeansMap[e[1]] = trCycleMeansMap[e[0]]
                        change = True


    length = dict()
    for n in gr.nodes():
        length[n] = None
    length[rootnode] = 0.0

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
                    if significantlySmaller(length[e[1]], newLength):
                        length[e[1]] = newLength
                        change = True

    # return the path lengths and the transitive cycle means
    return _normalizeDict(length), trCycleMeansMap



def mpEigenVectors(M: TMPMatrix) -> tuple[list[tuple[TMPVector,float]],list[tuple[TMPVector,TMPVector]]]:
    '''
    Compute the eigenvectors of a square matrix.
    Return a pair of a list of eigenvector and a list of generalized eigenvectors.
    '''
    # compute the precedence graphs
    # compute the strongly connected components and their cycle means
    # for each scc, determine a (generalized) eigenvector

    def _isRegularEigenValue(evv):
        values = set()
        for n in evv:
            if evv[n] is not None:
                values.add(evv[n])
        return len(values) <= 1

    def _asEigenValue(evv)->float:
        for n in evv:
            if evv[n] is not None:
                return evv[n]
        raise MPException("Eigenvalue cannot be minus infinity.")

    def _asGeneralizedEigenValue(evv):
        evl = []
        for n in gr.nodes():
            evl.append(evv[n])
        return evl

    gr = mpMatrixToPrecedenceGraph(M)

    # compute the strongly connected components of the precedence graph
    sccs = pyga.mutual_accessibility(gr)
    
    # keep lists of cycleMeans and subgraphs of the SCCs
    cycleMeans = []
    subgraphs = []
    
    # count the SCCs with k
    k = 0
    
    # a map from nodes of the precedence graph to their SCC index
    sccMap = dict()
    # map from SCC index to a node of the SCC
    sccMapInv = dict()
    # a list such that item k is a critical node of SCC k
    criticalNodes = list()
    # a map from nodes to the cycle mean of their SCC
    cycleMeansMap = dict()

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
            if trCycleMeans[sccMap[e[1]]] is None or trCycleMeans[sccMap[e[0]]] > trCycleMeans[sccMap[e[1]]]:
                change = True
                trCycleMeans[sccMap[e[1]]] = trCycleMeans[sccMap[e[0]]]

    # two lists to keep the results
    eigenVectors: list[tuple[TMPVector,float]] = []
    genEigenVectors: list[tuple[TMPVector,TMPVector]] = []
    
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

def mpPrecedenceGraph(M, labels):
    return mpMatrixToPrecedenceGraph(M, labels)

def mpPrecedenceGraphGraphviz(M, labels):
    gr = mpMatrixToPrecedenceGraph(M, labels)
    return weightedGraphToGraphViz(gr)

def mpStarClosure(M):
        return starClosure(M)

def mpConvolution(s, t):
    res = []
    l = min(len(s), len(t))
    for k in range(l):
        v = reduce(lambda mx, n: MP_MAX(mx, MP_PLUS(s[n], t[k-n])), range(k+1), MP_MINUSINFINITY)
        res.append(v)
    return res

def mpMaxEventSequences(es1, es2):
    return mpMaxVectors(es1, es2)


def mpMaxVectorSequences(vs1: TMPVectorList, vs2: TMPVectorList) -> TMPVectorList:
    return [mpMaxVectors(vs1[k], vs2[k]) for k in range(min(len(vs1), len(vs2)))]

def mpSplitSequence(seq, n):
    return [seq[k::n] for k in range(n)] 

def mpMergeSequences(seqs):
    res =  zip(*seqs)
    # print(seqs)
    # print(res)
    return res

def mpDelay(seq, n):
    return ([MP_MINUSINFINITY] * n) + seq

def mpScale(seq, c)->TTimeStampList:
    return [MP_PLUS(v, c) for v in seq]

