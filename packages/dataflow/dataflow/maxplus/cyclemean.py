'''Graph related algorithms.'''

# pygraph library from:
# https://pypi.org/project/python-graph/
# https://github.com/Shoobx/python-graph

from fractions import Fraction
from typing import Any, Dict, List, Set, Tuple, Union
import pygraph.classes.digraph  as pyg

class CycleMeanException(Exception):
    '''Exceptions in this library.'''

class Tree:
    '''Weighted Tree for cycle mean algorithms.'''

    _nodes: Set[Any]
    _root: Any
    _path_lengths: Dict[Any,Fraction]
    _parents: Dict[Any,Any]
    _children: Dict[Any,Set[Any]]

    def __init__(self, n: Any):
        self._nodes = {n}
        self._root = n
        self._path_lengths = {}
        self._path_lengths[n] = Fraction(0.0)
        self._parents = {}
        self._children = {}
        self._parents[n] = None
        self._children[n] = set()

    def path_length_of(self, n: Any)->Fraction:
        '''Return the node's path length.'''
        return self._path_lengths[n]

    def parent_of(self, n: Any)->Any:
        '''Retrieve the parent in the tree.'''
        return self._parents[n]

    def contains_node(self, n: Any)->bool:
        '''Check if the node is part of the tree.'''
        return n in self._nodes

    def is_reflexive_predecessor(self, n1: Any, n2: Any)->bool:
        ''' Check if n1 is a (reflexive) predecessor of n2 '''
        n = n2
        while n is not None:
            if n == n1:
                return True
            n = self._parents[n]
        return False

    def add_node(self, n: Any, parent: Any, dist: Fraction):
        '''Add node to parent at given distance.'''
        self._nodes.add(n)
        self._parents[n] = parent
        self._children[n] = set()
        self._children[parent].add(n)
        self._path_lengths[n] = self._path_lengths[parent] + dist

    def modify_parent(self, n: Any, new_parent: Any, dist: Fraction):
        '''Move node from tree to a different parent at given distance from the new parent.'''

        def _update_descendants(nd: Any, offset: Fraction):
            for m in self._children[nd]:
                _update_descendants(m, offset)
            self._path_lengths[nd] += offset

        if n == self._root:
            raise CycleMeanException('Cannot change root.')
        offset = self._path_lengths[new_parent] + dist - self._path_lengths[n]
        _update_descendants(n, offset)
        self._children[self._parents[n]].remove(n)
        self._parents[n] = new_parent
        self._children[new_parent].add(n)

def maximum_cycle_mean(gr: pyg.digraph)->Union[Tuple[None,None,None],Tuple[Fraction,Tree,Any]]:
    '''
    given a strongly connected graph with at least one cycle
    find the maximum cycle mean and a longest path spanning tree.
    Returns a tuple with the maximum cycle mean, spanning tree, and a node on the critical cycle.
    '''

    def _new_lambda(s_tree: Tree, leaf: Any, nn: Any)->Fraction:
        # compute cycle mean of the cycle leaf through parents up to nn and back to leaf
        # print('newLambda from {} to {}'.format(nn, leaf))
        if leaf == nn:
            # print('self-edge on {} weight: {}'.format(nn, gr.edge_weight((nn, nn))))
            return gr.edge_weight((nn, nn))

        cycle_length: Fraction = gr.edge_weight((leaf, nn))
        cycle_steps: int = 1
        nd = leaf
        pn = s_tree.parent_of(nd)
        while nd != nn:
            # print('adding edge {} with weight: {}'.format((pn, nd), gr.edge_weight((pn, nd))))
            cycle_length += gr.edge_weight((pn, nd))
            cycle_steps += 1
            nd = pn
            pn = s_tree.parent_of(nd)

        # print('cycleLength / cycleSteps: {}/{}'.format(cycleLength, cycleSteps))
        return cycle_length / cycle_steps

    # bail if the graph does not have any edges
    gr_edges = list(gr.edges())
    if len(gr_edges) == 0:
        return None, None, None

    # start with lambda = undefined (lower bound) and increase whenever a cycle mean
    # larger than lambda is found
    lower_bound_cycle_mean: Fraction = Fraction(gr.edge_weight(gr_edges[0]))
    for e in gr_edges:
        if gr.edge_weight(e) < lower_bound_cycle_mean:
            lower_bound_cycle_mean = gr.edge_weight(e)
    c_lambda = lower_bound_cycle_mean - Fraction(1)

    # create a longest path weighted spanning tree without positive cycles.

    # - initialize tree with one node
    n_root = gr.nodes()[0]

    # find a node on the critical cycle
    critical_node: Any = None

    restart: bool = True

    # initialize spTree before the loop to satisfy the type checker.
    sp_tree: Tree= Tree(n_root)

    while restart:

        sp_tree = Tree(n_root)
        leaves: List[Any] = [n_root]
        restart: bool = False

        # while there are fresh leaves:
        while len(leaves)>0 and not restart:
            leaf = leaves.pop(0)
            # print('Leaf: {}'.format(leaf))
            # for all outgoing edges:
            nn: Any # digraph node
            for nn in gr.neighbors(leaf):
                # compute their path length
                path_length: Fraction = sp_tree.path_length_of(leaf) + \
                    Fraction(gr.edge_weight((leaf, nn))) - c_lambda
                # if it is already in the tree
                if sp_tree.contains_node(nn):
                    tree_path_length = sp_tree.path_length_of(nn)
                    # with equal or longer path length continue, otherwise
                    if tree_path_length< path_length:
                        # it is already in the tree with smaller path length
                        # check if it is a predecessor of the current node
                        if sp_tree.is_reflexive_predecessor(nn, leaf):
                            # compute new lambda and cycle and start again
                            c_lambda = _new_lambda(sp_tree, leaf, nn)
                            critical_node = nn
                            restart = True
                        else:
                            # it is not a predecessor, but on a side branch, make it a
                            # descendant of the current node and adapt all path lengths
                            sp_tree.modify_parent(nn, leaf, gr.edge_weight((leaf, nn)) - c_lambda)
                            leaves.append(nn)
                else:
                    # nn is new to the tree
                    sp_tree.add_node(nn, leaf, gr.edge_weight((leaf, nn)) - c_lambda)
                    leaves.append(nn)

    # - if we have completed the tree return the last lambda, last cycle and the
    #   spanning tree with asap times.

    return c_lambda, sp_tree, critical_node
