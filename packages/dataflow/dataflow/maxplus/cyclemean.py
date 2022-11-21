# pygraph library from:
# https://pypi.org/project/python-graph/
# https://github.com/Shoobx/python-graph

from fractions import Fraction
from typing import Any, Dict, List, Set, Tuple, Union
import pygraph.classes.digraph  as pyg

class Tree:

    _nodes: Set[Any]
    _root: Any
    _pathLengths: Dict[Any,Fraction]
    _parents: Dict[Any,Any]
    _children: Dict[Any,Set[Any]]

    def __init__(self, n: Any):
        self._nodes = {n}
        self._root = n
        self._pathLengths = dict()
        self._pathLengths[n] = Fraction(0.0)
        self._parents = dict()
        self._children = dict()
        self._parents[n] = None
        self._children[n] = set()

    def pathLengthOf(self, n: Any)->Fraction:
        return self._pathLengths[n]

    def parentOf(self, n: Any)->Any:
        return self._parents[n]

    def containsNode(self, n: Any)->bool:
        return n in self._nodes

    def isReflexivePredecessor(self, n1: Any, n2: Any)->bool:
        ''' Check if n1 is a (reflexive) predecessor of n2 '''
        n = n2
        while n is not None:
            if n == n1:
                return True
            n = self._parents[n]
        return False

    def addNode(self, n: Any, parent: Any, dist: Fraction):
        '''Add node to parent at given distance.'''
        self._nodes.add(n)
        self._parents[n] = parent
        self._children[n] = set()
        self._children[parent].add(n)
        self._pathLengths[n] = self._pathLengths[parent] + dist
        
    def modifyParent(self, n: Any, newParent: Any, dist: Fraction):
        '''Move node from tree to a different parent at given distance from the new parent.'''
        
        def _updateDescendants(nd: Any, offset: Fraction):
            for m in self._children[nd]:
                _updateDescendants(m, offset)
            self._pathLengths[nd] += offset
        
        if n == self._root:
            raise Exception('Cannot change root.')
        offset = self._pathLengths[newParent] + dist - self._pathLengths[n]
        _updateDescendants(n, offset)
        self._children[self._parents[n]].remove(n)
        self._parents[n] = newParent
        self._children[newParent].add(n)


# numerical comparisons

# NumericalEpsilon = 1e-8

# def significantlySmaller(x: Fraction, y: Fraction)->bool:
#     return y-x > NumericalEpsilon 

def maximumCycleMean(gr: pyg.digraph)->Union[Tuple[None,None,None],Tuple[Fraction,Tree,Any]]:
    ''' 
    given a strongly connected graph with at least one cycle
    find the maximum cycle mean and a longest path spanning tree.
    Returns a tuple with the maximum cycle mean, spanning tree, and a node on the critical cycle.
    '''

    def _newLambda(sTree: Tree, leaf: Any, nn: Any)->Fraction:
        # compute cycle mean of the cycle leaf through parents up to nn and back to leaf
        # print('newLambda from {} to {}'.format(nn, leaf))
        if leaf == nn:
            # print('self-edge on {} weight: {}'.format(nn, gr.edge_weight((nn, nn))))
            return gr.edge_weight((nn, nn))

        cycleLength: Fraction = gr.edge_weight((leaf, nn))
        cycleSteps: int = 1
        nd = leaf
        pn = sTree.parentOf(nd)
        while nd != nn:    
            # print('adding edge {} with weight: {}'.format((pn, nd), gr.edge_weight((pn, nd))))
            cycleLength += gr.edge_weight((pn, nd))
            cycleSteps += 1
            nd = pn
            pn = sTree.parentOf(nd)
        
        # print('cycleLength / cycleSteps: {}/{}'.format(cycleLength, cycleSteps))
        return cycleLength / cycleSteps

    # bail if the graph does not have any edges
    grEdges = list(gr.edges())
    if(len(grEdges) == 0):
        return None, None, None

    # start with lambda = undefined (lower bound) and increase whenever a cycle mean larger than lambda is found
    lowerBoundCycleMean: Fraction = Fraction(gr.edge_weight(grEdges[0]))
    for e in grEdges:
        if gr.edge_weight(e) < lowerBoundCycleMean:
            lowerBoundCycleMean = gr.edge_weight(e)
    cLambda = lowerBoundCycleMean - Fraction(1)
    
    # create a longest path weighted spanning tree without positive cycles.

    # - initialize tree with one node
    nRoot = gr.nodes()[0]

    # find a node on the critical cycle
    criticalNode: Any = None

    restart: bool = True
    
    # initialize spTree before the loop to satisfy the type checker.
    spTree: Tree= Tree(nRoot)
    
    while restart:

        spTree = Tree(nRoot)
        leaves: List[Any] = [nRoot]
        restart: bool = False
    
        # while there are fresh leaves:
        while len(leaves)>0 and not restart:
            leaf = leaves.pop(0)
            # print('Leaf: {}'.format(leaf))
            # for all outgoing edges:
            nn: Any # digraph node
            for nn in gr.neighbors(leaf):
                # compute their path length
                pathLength: Fraction = spTree.pathLengthOf(leaf) + Fraction(gr.edge_weight((leaf, nn))) - cLambda 
                # if it is already in the tree
                if spTree.containsNode(nn):
                    treePathLength = spTree.pathLengthOf(nn)
                    # with equal or longer path length continue, otherwise
                    if treePathLength< pathLength:
                        # it is already in the tree with smaller path length
                        # check if it is a predecessor of the current node
                        if spTree.isReflexivePredecessor(nn, leaf):
                            # compute new lambda and cycle and start again
                            cLambda = _newLambda(spTree, leaf, nn)
                            criticalNode = nn
                            restart = True
                        else:
                            # it is not a predecessor, but on a side branch, make it a descendant of the current node 
                            # and adapt all path lengths
                            spTree.modifyParent(nn, leaf, gr.edge_weight((leaf, nn)) - cLambda)       
                            leaves.append(nn)
                else:
                    # nn is new to the tree
                    spTree.addNode(nn, leaf, gr.edge_weight((leaf, nn)) - cLambda)
                    leaves.append(nn)

    # - if we have completed the tree return the last lambda, last cycle and the spanning tree with asap times.

    return cLambda, spTree, criticalNode