# pygraph library from:
# https://pypi.org/project/python-graph/
# https://github.com/Shoobx/python-graph

import pygraph.classes.digraph  as pyg


class Tree:

    def __init__(self, n):
        self._nodes = {n}
        self._root = n
        self._pathLengths = dict()
        self._pathLengths[n] = 0.0
        self._parents = dict()
        self._children = dict()
        self._parents[n] = None
        self._children[n] = set()

    def pathLengthOf(self, n):
        return self._pathLengths[n]

    def parentOf(self, n):
        return self._parents[n]

    def containsNode(self, n):
        return n in self._nodes

    def isReflexivePredecessor(self, n1, n2):
        ''' Check if n1 is a (reflexive) predecessor of n2 '''
        n = n2
        while n is not None:
            if n == n1:
                return True
            n = self._parents[n]
        return False

    def addNode(self, n, parent, dist):
        self._nodes.add(n)
        self._parents[n] = parent
        self._children[n] = set()
        self._children[parent].add(n)
        self._pathLengths[n] = self._pathLengths[parent] + dist
        

    def modifyParent(self, n, newParent, dist):
        
        def _updateDescendants(nd, offset):
            for m in self._children[nd]:
                _updateDescendants(m, offset)
            self._pathLengths[nd] += offset
        
        if n == self._root:
            raise Exception('Should not change root?')
        offset = self._pathLengths[newParent] + dist - self._pathLengths[n]
        _updateDescendants(n, offset)
        self._children[self._parents[n]].remove(n)
        self._parents[n] = newParent
        self._children[newParent].add(n)


# numerical comparisons

NumericalEpsilon = 1e-8

def significantlySmaller(x, y):
    return y-x > NumericalEpsilon 

def maximumCycleMean(gr):
    ''' 
    given a strongly connected graph with at least one cycle
    find the maximum cycle mean and a longest path spanning tree
    '''
    def _newLambda(stree, leaf, nn, pathLength, lmbda):
        # compute cycle mean of the cycle leaf through parents up to nn and back to leaf
        # print('newLambda from {} to {}'.format(nn, leaf))
        if leaf == nn:
            # print('self-edge on {} weight: {}'.format(nn, gr.edge_weight((nn, nn))))
            return gr.edge_weight((nn, nn))

        cycleLength = gr.edge_weight((leaf, nn))
        cycleSteps = 1
        nd = leaf
        pn = stree.parentOf(nd)
        while nd != nn:    
            # print('adding edge {} with weight: {}'.format((pn, nd), gr.edge_weight((pn, nd))))
            cycleLength += gr.edge_weight((pn, nd))
            cycleSteps += 1
            nd = pn
            pn = stree.parentOf(nd)
        
        # print('cycleLength / cycleSteps: {}/{}'.format(cycleLength, cycleSteps))
        return cycleLength / cycleSteps

    grEdges = list(gr.edges())
    if(len(grEdges) == 0):
        return None, None, None

    # start with lambda = undefined and increase whenever a cycle mean larger than lambda is found
    lowerBoundCycleMean = gr.edge_weight(grEdges[0])
    for e in grEdges:
        if gr.edge_weight(e) < lowerBoundCycleMean:
            lowerBoundCycleMean = gr.edge_weight(e)
    lmbd = lowerBoundCycleMean - 1.0
    
    # create a longest  path weighted spanning tree without positive cycles.

    # - initialize tree with one node
    nroot = gr.nodes()[0]

    # find a node on the critical cycle
    criticalNode = None

    restart = True

    while restart:

        # print('Trying lambda = {}'.format(lmbd))
        stree = Tree(nroot)
        leaves = [nroot]
        restart = False
    
        # while there are fresh leaves:
        while len(leaves)>0 and not restart:
            leaf = leaves.pop(0)
            # print('Leaf: {}'.format(leaf))
            # for all outgoing edges:
            for nn in gr.neighbors(leaf):
                # compute their path length
                pathLength = stree.pathLengthOf(leaf) + gr.edge_weight((leaf, nn)) - lmbd 
                # if it is already in the tree
                if stree.containsNode(nn):
                    treePathLength = stree.pathLengthOf(nn)
                    # with equal or longer path length continue, otherwise
                    if significantlySmaller(treePathLength, pathLength):
                        # it is already in the tree with smaller path length
                        # check if it is a predecessor of the current node
                        if stree.isReflexivePredecessor(nn, leaf):
                            # compute new lambda and cycle and start again
                            lmbd = _newLambda(stree, leaf, nn, pathLength, lmbd)
                            criticalNode = nn
                            restart = True
                        else:
                            # it is not a predecessor, but on a side branch, make it a descendant of the current node 
                            # and adapt all path lengths
                            stree.modifyParent(nn, leaf, gr.edge_weight((leaf, nn)) - lmbd)       
                else:
                    # nn is new to the tree
                    stree.addNode(nn, leaf, gr.edge_weight((leaf, nn)) - lmbd)
                    leaves.append(nn)

    # - if we have completed the tree return the last labmda, last cycle and the spanning tree with asap times.

    # print(lmbd)
    return lmbd, stree, criticalNode