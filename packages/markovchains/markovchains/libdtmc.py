from io import StringIO
from typing import AbstractSet, Any, Dict, List, Literal, Optional, Set, Tuple, Union

# pygraph library from:
# https://pypi.org/project/python-graph/
# https://github.com/Shoobx/python-graph

import pygraph.classes.digraph  as pyg
import pygraph.algorithms.accessibility as pyga
import pygraph.algorithms.searching as pygs
import numpy as np
import functools
import math
import random
import os
import time
from statistics import NormalDist
from markovchains.libdtmcgrammar import parseDTMCDSL
from markovchains.utils.utils import sortNames

class DTMCException(Exception):
    pass
class MarkovChain(object):

    # for floating point comparisons
    _epsilon = 1e-10

    # As a rule of thumb, a reasonable number of results are needed before certain calculations can be considered valid
    # this variable determines the number of results that are required for the markov simulation before the stop conditions
    # are checked. (Based on the law of strong numbers)
    _law: int = 30

    # states is a list of strings
    _states: List[str]
    # transitions maps states to another dictionary that maps target states to the probability of reaching that target state
    _transitions: Dict[str, Dict[str,float]]
    # initialProbabilities maps states to the probability of starting in a given state
    _initialProbabilities: Dict[str,float]
    # rewards is a dictionary that maps states to their rewards
    _rewards: Dict[str,float]
    # transitionMatrix holds the transition matrix of the Markov Chains if it is computed
    _transitionMatrix: Optional[Any]
    # initialProbabilityVector holds the initial probabilities in the form of a vector
    _initialProbabilityVector: Optional[Any]
    # Recurrent state for simulation run calibration
    _recurrent_state: Optional[str]


    def __init__(self):
        self._states = list()
        self._transitions = dict()
        self._initialProbabilities = dict()
        self._rewards = dict()
        self._transitionMatrix = None
        self._initialProbabilityVector = None
        self._recurrent_state = None
        # Set pseudo random seed based on os current time
        random_data = os.urandom(8)
        seed = int.from_bytes(random_data, byteorder="big")
        random.seed(seed)

    def asDSL(self, name: str, state_info:bool = True, reward_info:bool = True)->str:
        '''Return the model as a string of the domain-specific language.'''
        # keep track of the states that have been output
        statesOutput: Set[str] = set()
        # create string writer for the output
        output = StringIO()
        # write header
        output.write("markov chain {} {{\n".format(name))

        for tr in sorted(self.transitions()):
            if tr[0] in statesOutput or (not state_info and not reward_info):
                output.write("\t{} -- {} --> {}\n".format(tr[0], tr[1], tr[2]))
            else:
                statesOutput.add(tr[0])
                r = self.getReward(tr[0])
                i = self._initialProbabilities[tr[0]]
                if state_info and reward_info and r!=0:
                    output.write("\t{} [p: {}; r: {}] -- {} --> {}\n".format(tr[0], i, r, tr[1], tr[2]))
                elif state_info and (not reward_info or r==0):
                    output.write("\t{} [p: {}] -- {} --> {}\n".format(tr[0], i, tr[1], tr[2]))
                elif not state_info and (reward_info or r!=0):
                    output.write("\t{} [r: {}] -- {} --> {}\n".format(tr[0], r, tr[1], tr[2]))

        output.write("}\n")

        result = output.getvalue()
        output.close()
        return result

    @staticmethod
    def _smaller(x: float, y: float)->bool:
        '''Compare two floating point numbers allowing for rounding errors. x is smaller than y only if the difference is significant (more than epsilon)'''
        return y-x > MarkovChain._epsilon

    @staticmethod
    def _larger(x: float, y: float)->bool:
        '''Compare two floating point numbers allowing for rounding errors. x is larger than y only if the difference is significant (more than epsilon)'''
        return x-y > MarkovChain._epsilon

    @staticmethod
    def _isZero(x: float)->bool:
        '''Check if a floating point number is zero allowing for rounding errors. (x is considered zero if its absolute value is smaller than  epsilon)'''
        return abs(x) < MarkovChain._epsilon

    def addState(self, s: str):
        '''Add a state named s to the MC'''
        # check if it already exists or not
        if not s in self._states:
            self._states.append(s)

    def numberOfStates(self)->int:
        '''Return the number of states.'''
        return len(self._states)

    def states(self)->List[str]:
        '''Return the list of states.'''
        return self._states

    def sortStateNames(self):
        '''Sort the list of states.'''
        self._states = sortNames(self._states)
        self._initialProbabilityVector = None
        self._transitionMatrix = None
    
    def setInitialProbability(self, s: str, p: float):
        '''Set initial probability of state s to p.'''
        if not s in self._states:
            raise DTMCException('Unknown state')
        self._initialProbabilities[s] = p

    def setReward(self, s, r):
        '''Set reward of state s to r.'''
        if not s in self._states:
            raise DTMCException('Unknown state')
        self._rewards[s] = r

    def getReward(self, s: str)->float:
        '''Get reward of state s. Defaults to 0.0 if undefined.'''
        if not s in self._rewards:
            return 0.0
        return self._rewards[s]

    def setEdgeProbability(self, s: str, d: str, p: float):
        '''Set the probability of the transition from s to d to p.'''
        if not s in self._states or not d in self._states:
            raise DTMCException('Unknown state')
        if s not in self._transitions:
            self._transitions[s] = dict()
        self._transitions[s][d] = p
    
    def transitions(self)->Set[Tuple[str,float,str]]:
        '''Get the transitions of the dtmc as tuples (s, p, d). With source state s, destination state d and probability p.'''
        result = set()
        for i, srcstate in enumerate(self._transitions):
            for j, (dststate, p) in enumerate(self._transitions[srcstate].items()):
                result.add((srcstate, p, dststate))
        return result

    def addImplicitTransitions(self):
        '''Add the implicit transitions when the outgoing probabilities do not add up to one.'''
        # compute the transition matrix, which will have the implicit transition probabilities
        M: Any = self.transitionMatrix()
        # set all transitions according to the non-zero elements in the matrix
        N = len(self._states)
        for i in range(N):
            si = self._states[i]
            for j in range(N):
                sj = self._states[j]
                if not MarkovChain._isZero(M[i][j]):
                    # add the transition if it does not yet exist
                    if not si in self._transitions:
                        self.setEdgeProbability(si, sj, M[i][j])
                    else:
                        if not sj in self._transitions[si]:
                            self.setEdgeProbability(si, sj, M[i][j])

    def _completeTransitionMatrix(self):
        '''Complete the transition matrix with missing/implicit transitions.'''
        if self._transitionMatrix is None:
            raise DTMCException("Transition matrix is not yet initialized.")
        # ensure that all rows add up to 1.
        # compute the row sums
        sums = np.sum(self._transitionMatrix, axis=1)
        for n in range(len(self._states)):
            # if the row n sum is smaller than 1
            if MarkovChain._smaller(sums.item(n), 1.0):
                # try to add the missing probability mass on a self-loop on n, if it is not specified (is zero)
                if MarkovChain._isZero(self._transitionMatrix.item((n,n))):
                    self._transitionMatrix.itemset((n,n), 1.0 - sums.item(n))
                else:
                    # cannot interpret it as an implicit transition
                    raise DTMCException("probabilities do not add to one")
            else:
                if MarkovChain._larger(sums.item(n), 1.0):
                    raise DTMCException("probability mass is larger than one")

    def transitionMatrix(self)->Any:
        '''Computes and returns the transition matrix of the MC.'''
        N = len(self._states)
        self._transitionMatrix = np.zeros([N, N])
        row = 0
        for ss in self._states:
            if ss in self._transitions:
                col = 0
                for sd in self._states:
                    if sd in self._transitions[ss]:
                        self._transitionMatrix.itemset((row, col), self._transitions[ss][sd])
                    col += 1
            row += 1
        self._completeTransitionMatrix()
        return self._transitionMatrix

    def initialProbabilitySpecified(self, s: str)->bool:
        '''Return if state s has a specified initial probability.'''
        return s in self._initialProbabilities

    def rewardSpecified(self, s: str)->bool:
        '''Return if state s has a specified reward.'''
        return s in self._rewards

    def completeInitialProbabilityVector(self):
        '''Complete the vector of initial probabilities with implicit probabilities.'''
        if self._initialProbabilityVector is None:
            raise DTMCException("Initial probability vector is not yet initialized.")
        # ensure that the initial probabilities add up to 1.
        sum = np.sum(self._initialProbabilityVector, axis=0)
        if MarkovChain._larger(sum, 1.0):
            raise DTMCException("probability is larger than one")
        if MarkovChain._smaller(sum, 1.0):
            K = [self.initialProbabilitySpecified(s) for s in self._states].count(False)
            if K == 0:
                raise DTMCException("probability mass is smaller than one")
            remainder = (1.0 - sum) / K
            k = 0
            for s in self._states:
                if not self.initialProbabilitySpecified(s):
                    self._initialProbabilityVector.itemset(k, remainder)
                k += 1

    def completeInitialProbabilities(self):
        '''Complete the initial probabilities.'''
        # ensure that the initial probabilities add up to 1.
        sum = functools.reduce(lambda a,b : a+b, self._initialProbabilities.values(), 0.0)
        if MarkovChain._larger(sum, 1.0):
            raise DTMCException("initial probability mass is larger than one")
        if MarkovChain._smaller(sum, 1.0):
            K = [self.initialProbabilitySpecified(s) for s in self._states].count(False)
            if K == 0:
                raise DTMCException("initial probability mass is smaller than one")
            remainder = (1.0 - sum) / K
            k = 0
            for s in self._states:
                if not self.initialProbabilitySpecified(s):
                    self.setInitialProbability(s, remainder)
                k += 1

    def completeRewards(self):
        '''Complete the implicit rewards to zero.'''
        # Initialize reward zero if not defined in dtmc model
        for s in self._states:
            if not self.rewardSpecified(s):
                self.setReward(s, 0.0)

    def initialProbabilityVector(self)->Any:
        '''Determine and return the initial probability vector.'''
        N = len(self._states)
        self._initialProbabilityVector = np.zeros([N])
        k = 0
        for s in self._states:
            if s in self._initialProbabilities:
                self._initialProbabilityVector.itemset(k, self._initialProbabilities[s])
            k += 1
        self.completeInitialProbabilityVector()
        return self._initialProbabilityVector

    def rewardVector(self)->Any:
        '''Return reward vector.'''
        res = np.empty(len(self._states))
        k = 0
        for s in self._states:
            res[k] = self.getReward(s)
            k += 1
        return res

    def rewardForDistribution(self, d: Any)->float:
        '''Return expected reward for a given distribution.'''
        result = 0.0
        for k in range(self.numberOfStates()):
            result += d[k] * self.getReward(self._states[k])
        return result

    def executeSteps(self, N: int)->Any:
        '''Perform N steps of the chain, return array of N+1 distributions, starting with the initial distribution and distributions after N steps.'''
        P = self.transitionMatrix()
        pi = self.initialProbabilityVector()
        result = np.empty([N+1, self.numberOfStates()])
        for k in range(N+1):
            result[k] = pi
            pi = np.matmul(pi, P)
        return result


    def _computeCommunicationGraph(self)->pyg.digraph:
        '''Return the communication graph corresponding to the Markov Chain.'''
        gr = pyg.digraph()
        gr.add_nodes(self._states)
        for s in self._transitions:
            for t in self._transitions[s]:
                if not MarkovChain._isZero(self._transitions[s][t]):
                    gr.add_edge((s, t))
        return gr

    def _computeReverseCommunicationGraph(self)->pyg.digraph:
        '''Return the communication graph corresponding to the reversed transitions of the Markov Chain.'''
        gr = pyg.digraph()
        gr.add_nodes(self._states)
        for s in self._transitions:
            for t in self._transitions[s]:
                if not MarkovChain._isZero(self._transitions[s][t]):
                    gr.add_edge((t, s))
        return gr

    def communicatingClasses(self)->Set[AbstractSet[str]]:
        '''Determine the communicating classes of the dtmc. Returns Set of sets of states of the dtmc.'''
        gr = self._computeCommunicationGraph()
        return set([frozenset(s) for s in pyga.mutual_accessibility(gr).values()])

    def classifyTransientRecurrentClasses(self)->Tuple[Set[AbstractSet[str]],Set[AbstractSet[str]]]:
        '''Classify the states into transient and recurrent. Return a pair with transient classes and recurrent classes.'''
        cClasses = self.communicatingClasses()
        rClasses = cClasses.copy()
        stateMap = dict()
        for c in cClasses:
            for s in c:
                stateMap[s] = c

        # remove all classes with outgoing transitions
        for s in self._transitions:
            if stateMap[s] in rClasses:
                for t in self._transitions[s]:
                    if stateMap[t] != stateMap[s]:
                        if stateMap[s] in rClasses:
                            rClasses.remove(stateMap[s])

        return cClasses, rClasses

    def classifyTransientRecurrent(self)->Tuple[Set[str],Set[str]]:
        '''Classify states into transient and recurrent.'''
        _, rClasses = self.classifyTransientRecurrentClasses()

        # collect all recurrent states
        rStates = set()
        for c in rClasses:
            rStates.update(c)

        # remaining states are transient
        tStates = set(self._states).difference(rStates)

        return tStates, rStates

    def classifyPeriodicity(self)->Dict[str,int]:
        '''Determine periodicity of states. Returns a dictionary mapping state to periodicity.'''

        def _cycleFound(k:str):
            nonlocal nodeStack, cyclesFound
            i = nodeStack.index(k)
            cyclesFound.add(frozenset(nodeStack[i:len(nodeStack)]))

        def _exploreCycles(m: str):
            exploredNodes.add(m)
            nodeStack.append(m)
            for k in gr.neighbors(m):
                if k in nodeStack:
                    _cycleFound(k)
                else:
                    _exploreCycles(k)
            nodeStack.pop(len(nodeStack)-1)

        self.addImplicitTransitions()

        gr = self._computeCommunicationGraph()
        # perform exploration for all states
        cyclesFound:Set[AbstractSet[str]] = set()
        exploredNodes: Set[str] =set()
        nodesToExplore: List[str] = list(gr.nodes())
        nodeStack: List[str] = list()
        while len(nodesToExplore) > 0:
            n: str = nodesToExplore.pop(0)
            if not n in exploredNodes:
                _exploreCycles(n)

        # compute periodicities of the recurrent states
        # periodicity is gcd of the length of all cycles reachable from the state
        _, rStates = self.classifyTransientRecurrent()

        per: Dict[str,int] = dict()
        for c in cyclesFound:
            cl = len(c)
            for s in c:
                if s in rStates:
                    if not s in per:
                        per[s] = cl
                    else:
                        per[s] = math.gcd(cl, per[s])

        commClasses = self.communicatingClasses()
        for cl in commClasses:
            s = next(iter(cl))
            if s in rStates:
                p = per[s]
                for s in cl:
                    p = math.gcd(p, per[s])
                for s in cl:
                    per[s] = p

        return per

    def determineMCType(self)->Literal['ergodic unichain','non-ergodic unichain','ergodic non-unichain','non-ergodic non-unichain']:
        '''Return the type of the MC.
        A class that is both recurrent and aperiodic is called an ergodic class.
        A Markov chain having a single class of communicating states is called an irreducible Markov chain. Notice that this class of states is necessarily recurrent. In case this class is also aperiodic, i.e. if the class is ergodic, the chain is called an ergodic Markov chain.
        #A Markov chain that contains a single recurrent class in addition to zero or more transient classes, is called a unichain. In case the recurrent class is ergodic, we speak about an ergodic unichain. A unichain visits its transients states a finite number of times, after which the chain enters the unique class of recurrent states in which it remains for ever.
        '''

        _, rClasses =  self.classifyTransientRecurrentClasses()
        per = self.classifyPeriodicity()

        isUnichain = len(rClasses) == 1
        eClasses = [c for c in rClasses if per[next(iter(c))] == 1]

        if isUnichain:
            if len(eClasses) > 0:
                return 'ergodic unichain'
            else:
                return 'non-ergodic unichain'
        if len(rClasses) == len(eClasses):
            return 'ergodic non-unichain'

        return 'non-ergodic non-unichain'

    def _hittingProbabilities(self, targetState: str)->Tuple[List[str],Optional[Any],Dict[str,int],Dict[str,int],Dict[str,float]]:
        '''Determine the hitting probabilities to hit targetState. Returns a tuple with:
        - rs: the list of states from which the target state is reachable
        - ImEQ: the matrix of the matrix equation
        - rIndex: index numbers of the states from rs in the equation
        - pIndex, index of all states
        - res: the hitting probabilities 
        '''

        def _statesReachableFrom(s: str)->List[str]:
            _, pre, _ = pygs.depth_first_search(gr, root=s)
            return pre

        # determine the set of states from which targetState is reachable
        gr = self._computeReverseCommunicationGraph()
        rs = _statesReachableFrom(targetState)
        rs = list(rs)
        # give each of the states an index in map rIndex
        rIndex:Dict[str,int] = dict()
        for k in range(len(rs)):
            rIndex[rs[k]] = k

        # determine transition matrix and state indices
        P = self.transitionMatrix()
        pIndex:Dict[str,int] = dict()
        for k in range(len(self._states)):
            pIndex[self._states[k]] = k
        # jp is index of the target state
        jp = pIndex[targetState]

        ImEQ = None
        solX = None

        if  len(rs) > 0:

            # determine the hitting prob equations
            # Let us fix a state j ∈ S and define for each state i in rs a corresponding variable x_i (representing the hit probability f_ij ). Consider the system of linear equations defined by
            # x_i = Pij + Sum _ k∈S\{j} P_ik x_k
            # solve the equation: x = EQ x + Pj
            # solve the equation: (I-EQ) x = Pj

            N = len(rs)
            # initialize matrix I-EQ from the equation, and vector Pj
            ImEQ = np.identity(N)
            pj = np.zeros([N])
            # for all equations (rows of the matrix)
            for i in range(N):
                ip = pIndex[rs[i]]
                pj[i] = P[ip][jp]
                # for all variables in the summand
                for k in range(N):
                    # note that the sum excludes the target state!
                    if rs[k] != targetState:
                        kp = pIndex[rs[k]]
                        # determine row i, column k
                        ImEQ[i][k] -= P[ip][kp]

            # solve the equation x = inv(I-EQ)*Pj
            solX = np.matmul(np.linalg.inv(ImEQ), pj)

        # set all hitting probabilities to zero
        res: Dict[str,float] = dict()
        for s in self._states:
            res[s] = 0.0

        # fill the solutions from the equation
        for s in rs:
            solX: Any
            res[s] = solX[rIndex[s]]

        return rs, ImEQ, rIndex, pIndex, res


    def hittingProbabilities(self, targetState: str)->Dict[str,float]:
        '''Determine the hitting probabilities to hit targetState.'''

        _, _, _, _, res = self._hittingProbabilities(targetState)
        return res

    def rewardTillHit(self, targetState: str)->Dict[str,float]:
        '''Determine the expected reward until hitting targetState'''

        rs, ImEQ, rIndex, pIndex, f = self._hittingProbabilities(targetState)
        solX = None
        if  len(rs) > 0:

            # x_i = r(i) · fij + Sum k∈S\{j} P_ik x_k
            # solve the equation: x = EQ x + Fj
            # solve the equation: (I-EQ) x = Fj

            N = len(rs)
            fj = np.zeros([N])
            jp = pIndex[targetState]
            for i in range(N):
                si = rs[i]
                ip = pIndex[si]
                fj[i] = self.getReward(si) * f[si]


            # solve the equation x = inv(I-EQ)*Fj
            ImEQ: Any # is not None
            solX = np.matmul(np.linalg.inv(ImEQ), fj)

        # set fr to zero
        fr = dict()
        for s in self._states:
            fr[s] = 0.0

        # fill the solutions from the equation
        for s in rs:
            solX: Any # is not None
            fr[s] = solX[rIndex[s]]

        res = dict()
        for s in rs:
            res[s] = fr[s] / f[s]

        return res


    def _hittingProbabilitiesSet(self, targetStates: AbstractSet[str])->Tuple[Any,List[str],Optional[Any],Dict[str,int],Dict[str,int],Dict[str,float]]:
        '''Determine the hitting probabilities to hit a set targetStates. Returns a tuple with:
        
        - P: the transition matrix
        - rs: the list of states from which the target state is reachable
        - ImEQ: the matrix of the matrix equation
        - rIndex: index numbers of the states from rs in the equation
        - pIndex, index of all states
        - res: the hitting probabilities 
        '''

        def _statesReachableFrom(s: str)->List[str]:
            _, pre, _ = pygs.depth_first_search(gr, root=s)
            return pre

        # determine the set of states from which the set targetStates are reachable
        # not very efficient...
        gr = self._computeReverseCommunicationGraph()
        rs = set()
        for s in targetStates:
            rs.update(_statesReachableFrom(s))

        # exclude target states
        rs = rs.difference(targetStates)

        # fix an arbitrary order
        rs = list(rs)

        # make an index on rs
        rIndex = dict()
        for k in range(len(rs)):
            rIndex[rs[k]] = k

        # get transition matrix
        P = self.transitionMatrix()
        # make index on P matrix
        pIndex = dict()
        for k in range(len(self._states)):
            pIndex[self._states[k]] = k

        # determine the hitting prob equations
        # x_i = 0                                                           if i in S \ (rs U targetStates)
        # x_i = 1                                                           if i in targetStates
        # x_i = sum _ k in rs P_ik x_k + sum _ k in targetStates P_ik       if i in rs
        # equation for the third case, take first two cases as constants:
        # x_i = sum _ k in rs P_ik x_k + SP(i) = sum _ k in targetStates P_ik       for i in rs

        # solve the equation: x = EQ x + Pj
        # solve the equation: (I-EQ) x = Pj

        N = len(rs)
        ImEQ = np.identity(N)
        sp = np.zeros([N])
        for i in range(N):
            ip = pIndex[rs[i]]
            # compute the i-th element in vector SP
            for s in targetStates:
                sp[i] += P[ip][pIndex[s]]

            # compute the i-th row in matrix ImEQ
            for k in range(N):
                kp = pIndex[rs[k]]
                # determine row i, column k
                ImEQ[i][k] -= P[ip][kp]

        # solve the equation x = inv(I-EQ)*SP
        solX = np.matmul(np.linalg.inv(ImEQ), sp)

        # set all hitting probabilities to zero
        res = dict()
        for s in self._states:
            res[s] = 0.0

        # set all hitting probabilities in the target set to 1
        for s in targetStates:
            res[s] = 1.0

        # fill the solutions from the equation
        for s in rs:
            res[s] = solX[rIndex[s]]

        return P, rs, ImEQ, rIndex, pIndex, res

    def hittingProbabilitiesSet(self, targetStates: AbstractSet[str])->Dict[str,float]:
        '''Determine the hitting probabilities to hit a set targetStates.'''
        _, _, _, _, _, res = self._hittingProbabilitiesSet(targetStates)
        return res

    def rewardTillHitSet(self, targetStates: Set[str]):
        '''Determine the expected reward until hitting set targetStates.'''

        _, rs, ImEQ, rIndex, _, h = self._hittingProbabilitiesSet(targetStates)

        solX = None

        if  len(rs) > 0:

            # x_i = sum _ k in rs P_ik x_k  +  hh(i) = r(i) · h_i    for all i in rs

            # solve the equation: x = EQ x + H
            # solve the equation: (I-EQ) x = H

            N = len(rs)
            hh = np.zeros([N])
            for i in range(N):
                # compute the i-th element in vector H
                si = rs[i]
                hh[i] = self.getReward(si) * h[si]

            # solve the equation x = inv(I-EQ)*H
            ImEQ: Any
            solX = np.matmul(np.linalg.inv(ImEQ), hh)

        # set hr to zero
        hr = dict()
        for s in self._states:
            hr[s] = 0.0

        # fill the solutions from the equation
        for s in rs:
            solX: Any
            hr[s] = solX[rIndex[s]]

        res = dict()
        for s in targetStates:
            res[s] = 0.0
        for s in rs:
            res[s] = hr[s] / h[s]

        return res

    def _getSubTransitionMatrixIndices(self, indices: List[int])->Any:
        '''Return sub transition matrix consisting of the given list of indices.'''
        if self._transitionMatrix is None:
            raise DTMCException("Transition matrix has not been determined.")
        N = len(indices)
        res = np.zeros([N, N])
        for k in range(N):
            for m in range(N):
                res[k][m] = self._transitionMatrix[indices[k]][indices[m]]
        return res

    def _getSubTransitionMatrixClass(self, C: AbstractSet[str])->Tuple[Dict[str,int],Any]:
        '''Return an index for the states in C and a sub transition matrix for the class C.'''
        # get submatrix for a class C of states
        indices = sorted([self._states.index(s) for s in C])
        index = dict([(c, indices.index(self._states.index(c))) for c in C])
        return index, self._getSubTransitionMatrixIndices(indices)

    def limitingMatrix(self)->Any:
        '''Determine the limiting matrix of the dtmc.'''
        # formulate and solve balance equations for each of the  recurrent classes
        # determine the recurrent classes
        self.transitionMatrix()

        N = len(self._states)
        res = np.zeros([N, N])

        _, rClasses =  self.classifyTransientRecurrentClasses()

        # for each recurrent class:
        for c in rClasses:
            index, Pc = self._getSubTransitionMatrixClass(c)
            # a) solve the balance equations, pi P = pi I , pi.1 = 1
            #       pi (P-I); 1 = [0 1],
            M = len(c)
            PmI = np.subtract(Pc, np.eye(M))
            Q = np.matmul(PmI, np.matrix.transpose(PmI)) + np.ones([M,M])  # type: ignore numpy internal type issue
            QInv = np.linalg.inv(Q)
            pi = np.sum(QInv, 0)
            h = self.hittingProbabilitiesSet(c)
            # P(i,j) = h_i * pi j
            for sj in c:
                j = self._states.index(sj)
                for i in range(N):
                    if self._states[i] in c:
                        res[i][j] = pi[index[sj]]
                    else:
                        res[i][j] = h[self._states[i]] * pi[index[sj]]
        return res

    def limitingDistribution(self)->Any:
        '''Determine the limiting distribution.'''
        P = self.limitingMatrix()
        pi0 = self.initialProbabilityVector()
        return np.dot(pi0, P)

    def longRunReward(self)-> float:
        '''Determine the long-run expected reward.'''
        pi = self.limitingDistribution()
        r = self.rewardVector()
        return np.dot(pi, r)

    @staticmethod
    def fromDSL(dslString):

        factory = dict()
        factory['Init'] = lambda : MarkovChain()
        factory['AddState'] = lambda dtmc, s: (dtmc.addState(s), s)[1]
        factory['SetInitialProbability'] = lambda dtmc, s, p: dtmc.setInitialProbability(s, p)
        factory['SetReward'] = lambda dtmc, s, r: dtmc.setReward(s, r)
        factory['SetEdgeProbability'] = lambda dtmc, s, d, p: dtmc.setEdgeProbability(s, d, p)
        factory['SortNames'] = lambda dtmc: dtmc.sortStateNames()
        return parseDTMCDSL(dslString, factory)

    def __str__(self):
        return str(self._states)


    # ---------------------------------------------------------------------------- #
    # - Markovchain simulation                                                   - #
    # ---------------------------------------------------------------------------- #

    def setSeed(self, seed):
        # Initialize random generator to seed provided by the user
        random.seed(seed)

    def setRecurrentState(self, state):
        if self._isRecurrentState(state):
            self._recurrent_state = state
        else:
            self._recurrent_state = None

    def _markovSimulation(self, actions):
        # Set current state to None to indicate that the first state is not defined
        current_state = None
        stop = None

        # list for stopcondition status
        stop_conditions = [False] * len(actions)
        action_conditions = [None] * len(actions)

        # calculate random value for state transition using seed thats set in the beginning
        r = random.random()

        # Calculate random markov chain sequence
        n = 0
        # set probability count to zero
        p = 0.0

        # Determine initial state
        for s in self._initialProbabilities:
            p = p + self._initialProbabilities[s]
            if r < p:
                for i, action in enumerate(actions):
                    action_conditions[i] = action[0](n, s)
                    stop_conditions[i] = action[1](action_conditions[i])

                current_state = s
                n += 1

                break

        while not any(stop_conditions):
            # calculate random value for state transition
            r = random.random()
            # set probability count to zero
            p = 0.0
            # Look through all transition probabilities of current state.
            for s in self._transitions[current_state]:
                p = p + self._transitions[current_state][s]
                if r < p:
                    for i, action in enumerate(actions):
                        action_conditions[i] = action[0](n, s)
                        stop_conditions[i] = action[1](action_conditions[i])
                    current_state = s

                    break

            # next step in markovchain simulation
            n += 1

        # Determine stop condition (if it is added)
        for i, action in enumerate(actions):
            if len(action) > 2:
                if stop_conditions[i]:
                    stop = action[2]

        return n, stop

    def _cPointEstimate(self, confidence):
        return NormalDist().inv_cdf((1+confidence)/2)

    def _isRecurrentState(self, state):
        # recurrent state thats encountered in the random state_sequence dictionary
        _, rClasses = self.classifyTransientRecurrent()

        for recurrent in rClasses:
            if state == recurrent:
                return True

        return False

    def _cycleUpdate(self, state):
        # Find first recurrent state
        if self._recurrent_state == None:
            if self._isRecurrentState(state):
                self._recurrent_state = state
            else:
                return -1 # To prevent the simulation from exiting

        if self._recurrent_state == state:
            self.Em += 1
            self.El += self.l
            self.Er += self.r
            self.El2 += pow(self.l, 2)
            self.Er2 += pow(self.r, 2)
            self.Elr += self.l * self.r
            self.l = 0
            self.r = 0

        self.l += 1
        self.r += self._rewards[state]

        return self.Em

    def _cycleUpdateCezaro(self, state):
        # Find first recurrent state
        if self._recurrent_state == None:
            if self._isRecurrentState(state):
                self._recurrent_state = state
            else:
                return -1 # To prevent the simulation from exiting

        if self._recurrent_state == state:
            self.Em += 1
            self.El += self.l
            self.El2 += pow(self.l, 2)
            self.Er = np.add(self.Er, self.r)
            self.Er2 = np.add(self.Er2, [pow(i, 2) for i in self.r])
            self.Elr = np.add(self.Elr, [i*self.l for i in self.r])
            self.r = [0.0] * len(self._states)
            self.l = 0

        self.l += 1
        self.r[self._states.index(state)] += 1

        return self.Em

    def _pointEstU(self):
        if self.El != 0:
            self.u = self.Er/self.El
 
    def _pointEstSm(self):
        if (self.El != 0) and (self.Em != 0):
            self.Sm = math.sqrt(abs((self.Er2 - 2*self.u*self.Elr + pow(self.u,2)*self.El2)/self.Em))

    def _pointEstSmCezaro(self):
        if (self.El != 0) and (self.Em != 0):
            for i in range(len(self.Sm)):
                self.Sm[i] = math.sqrt(abs((self.Er2[i] - 2*self.u[i]*self.Elr[i] + pow(self.u[i],2)*self.El2)/self.Em))

    def _abError(self, con, n):
        # Run first MarkovChain._law times without checking abError
        if 0 <= n < MarkovChain._law:
            return -1.0

        if self.Em > 0:
            d = math.sqrt(self.Em) * (1/(self.Em)) * self.El
            if d != 0:
                return abs((con*self.Sm) / d)
            else:
                return -1.0
        else:
            return -1.0

    def _abErrorCezaro(self, con, n):
        # Run first MarkovChain._law times without checking abError
        abError = [-1.0] * len(self._states)

        if 0 <= n < MarkovChain._law:
            return abError

        if self.Em > 0:
            d = math.sqrt(self.Em) * (1/(self.Em)) * self.El
            for i in range(len(abError)):
                if d != 0:
                    abError[i] = abs((con*self.Sm[i]) / d)
                else:
                    abError[i] = -1.0
  
        return abError

    def _reError(self, con, n):
        # Run first 10 times without checking abError
        if 0 <= n < MarkovChain._law:
            return -1.0

        if self.Em > 0:
            d = (self.u * np.sqrt(self.Em) * (1/(self.Em)) * self.El) - (con*self.Sm)
            if d != 0:
                return abs((con*self.Sm) / d)
            else:
                return -1.0
        else:
            return -1.0

    def _reErrorCezaro(self, con, n):
        reError = [-1.0] * len(self._states)

        # Run first 10 times without checking abError
        if 0 <= n < MarkovChain._law:
            return reError
        
        for i in range(len(reError)):
            if self.Em > 0:
                d = (self.u[i] * np.sqrt(self.Em) * (1/(self.Em)) * self.El) - (con*self.Sm[i])
                if d != 0:
                    reError[i] = abs((con*self.Sm[i]) / d)
                else:
                    reError[i] = -1.0

        return reError

    def _lastStateReward(self, nr_of_steps, n, state):
        if n == nr_of_steps:
            self.y = self._rewards[state]

    def _hittingStateCount(self, n, state, nr_of_steps):
        if n == nr_of_steps:
            for i in range(len(self.state_count)):
                if state == self._states[i]:
                    self.state_count[i] += 1

    def _traceReward(self, condition, state, goal_state):
        # check state for goal state set or single goal state
        if isinstance(goal_state, list):
            valid = any(state == s for s in goal_state)
        else:
            valid = state == goal_state

        if condition:
            if valid:
                self.valid_trace = True
                return True
            else:
                self.sum_rewards += self._rewards[state]
                self.valid_trace = False
                return False
        else: # In case initial state is goal_state
            self.sum_rewards += self._rewards[state]
            self.valid_trace = False
            return False

    def markovTrace(self, rounds):
        # Declare empty list for states
        state_list = []
        self._markovSimulation([
            [lambda n, state: n, lambda c : c >= rounds], # Exit when n is number of rounds
            [lambda n, state: state_list.append(state), lambda c : False] # add state to list
        ])

        return state_list

    def longRunExpectedAverageReward(self, stop_conditions):
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        max_path_length = stop_conditions[3]
        nr_of_cycles = stop_conditions[4]
        seconds = stop_conditions[5]

        # Calculate the confidence point estimate with inverse normal distribution
        con = self._cPointEstimate(confidence)

        # Global variables
        self.l = 0 # Current cycle length
        self.r = 0 # Current cycle cumulative reward
        self.Em = -1 # Cycle count (-1 to subtract unfinished cycle beforehand)
        self.El = 0 # Sum of cycle lengths
        self.Er = 0 # Sum of cumulative rewards
        self.El2 = 0 # Sum of cycle length squared
        self.Er2 = 0 # Sum of cycle cumulative reward squared
        self.Elr = 0 # Sum of cycle product length and cycle
        self.u = 0 # Estimated mean
        self.Sm = 0 # Estimated variance

        # Save current time
        current_time = time.time()
        _, stop = self._markovSimulation([
            [lambda n, state: n, lambda c : 0 <= max_path_length <= c, "Maximum path length"], # Run until max path length has been reached
            [lambda n, state: time.time() - current_time, lambda c : 0 <= seconds <= c, "Timeout"], # Exit on time
            [lambda n, state: self._cycleUpdate(state), lambda c : 0 <= nr_of_cycles <= c, "Number of cycles"], # find first recurrent state
            [lambda n, state: self._pointEstU(), lambda c : False], # update point estimate of u
            [lambda n, state: self._pointEstSm(), lambda c : False], # update point estimate of Sm
            [lambda n, state: self._abError(con, n), lambda c : 0 <= c <= max_abError, "Absolute Error"], # update point estimate of Sm
            [lambda n, state: self._reError(con, n), lambda c : 0 <= c <= max_reError, "Relative Error"]
        ])

        # Compute absolute/relative error regardless of law of strong numbers
        abError = self._abError(con, MarkovChain._law)
        reError = self._reError(con, MarkovChain._law)

        # Compute confidence interval
        interval = [self.u - abError, self.u + abError]

        # Check reError
        if (interval[0] < 0 and interval[1] >= 0) or reError < 0:
            reError = None
        
        # Check abError
        if abError < 0: # Error value
            reError = None
            abError = None
            interval = [None, None]

        return interval, abError, reError, self.u, self.Em, stop

    def cezaroLimitDistribution(self, stop_conditions):
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        max_path_length = stop_conditions[3]
        nr_of_cycles = stop_conditions[4]
        seconds = stop_conditions[5]

        # Fetch number of states in Markov chain
        nr_of_states = self.numberOfStates()

        # Calculate the confidence point estimate with inverse normal distribution
        con = self._cPointEstimate(confidence)

        # Global variables
        self.l = 0 # Current cycle length
        self.Em = -1 # Cycle count (-1 to subtract unfinished cycle beforehand)
        self.El = 0 # Sum of cycle lengths
        self.El2 = 0 # Sum of cycle length squared
        self.r = [0.0] * nr_of_states # Current cycle cumulative reward
        self.Er = [0.0] * nr_of_states # Sum of cumulative rewards
        self.Er2 = [0.0] * nr_of_states # Sum of cycle cumulative reward squared
        self.Elr = [0.0] * nr_of_states # Sum of cycle product length and cycle
        self.u = [0.0] * nr_of_states # Estimated mean
        self.Sm = [0.0] * nr_of_states # Estimated variance

        state_list = []

        # Save current time
        current_time = time.time()
        _, stop = self._markovSimulation([
            [lambda n, state: n, lambda c : 0 <= max_path_length <= c, "Maximum path length"], # Run until max path length has been reached
            [lambda n, state: time.time() - current_time, lambda c : 0 <= seconds <= c, "Timeout"], # Exit on time
            [lambda n, state: self._cycleUpdateCezaro(state), lambda c : 0 <= nr_of_cycles <= c, "Number of cycles"], # find first recurrent state
            [lambda n, state: self._pointEstU(), lambda c : False], # update point estimate of u
            [lambda n, state: self._pointEstSmCezaro(), lambda c : False], # update point estimate of Sm
            [lambda n, state: self._abErrorCezaro(con, n), lambda c : 0 <= max(c) <= max_abError, "Absolute Error"], # Calcute smallest absolute error
            [lambda n, state: self._reErrorCezaro(con, n), lambda c : 0 <= max(c) <= max_reError, "Relative Error"], # Calcute smallest relative error
            [lambda n, state: state_list.append(state), lambda c : False] # Calcute smallest relative error
        ])

        abError = self._abErrorCezaro(con, MarkovChain._law)
        reError = self._reErrorCezaro(con, MarkovChain._law)
        interval = []
        for i in range(nr_of_states):
            # Compute confidence interval
            interval.append([self.u[i] - abError[i], self.u[i] + abError[i]])

            # Check reError
            if (interval[i][0] < 0 and interval[i][1] >= 0) or reError[i] < 0:
                reError[i] = None
            
            # Check abError
            if abError[i] < 0: # Error value
                reError[i] = None
                abError[i] = None
                interval[i] = [None, None]

        # Check if sum of distribution equals 1 with .4 float accuracy
        if not 0.9999 < sum(self.u) < 1.0001:
            self.u = None

        return self.u, interval, abError, reError, self.Em, stop

    def estimationExpectedReward(self, stop_conditions, nr_of_steps):
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]

        # confidence level
        c = self._cPointEstimate(confidence)

        interval = [0, 0]
        abError = -1.0
        reError = -1.0
        u = 0
        m = 0
        M2 = 0 # Welford's algorithm variable

        # There are in total four applicable stop conditions for this function
        actions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        stop_conditions = [False] * 4

        # Save current time
        current_time = time.time()
        while not any(stop_conditions):
            # Used in self_lastStateReward function to assign reward to y when hitting last state
            self.y = 0 
            self._markovSimulation([
                [lambda n, state: n, lambda c : 0 <= nr_of_steps <= c], # Exit when n is number of rounds
                [lambda n, state: time.time() - current_time, lambda c : 0 <= seconds <= c], # Exit on time
                [lambda n, state: self._lastStateReward(nr_of_steps, n, state), lambda c : False]
            ])

            # Execute Welford's algorithm to compute running standard derivation and mean
            m += 1
            d1 = self.y - u
            u += d1/m
            d2 = self.y - u
            M2 += d1 * d2
            Sm = np.sqrt(M2/m)

            # Compute absolute and relative errors
            abError = abs((c*Sm) / np.sqrt(m))
            d = u-abError
            if d != 0.0:
                reError = abs(abError/d)

            # interval calculation
            interval = [u - abError, u + abError]

            # Do not evaluate abError/reError in the first _law cycles:
            if m < MarkovChain._law and m != rounds: # (if rounds is less than 10 we still want an abError and reError)
                abError = -1.0
                reError = -1.0

            # Check stop conditions
            stop_conditions[0] = (0.0 <= abError <= max_abError)
            stop_conditions[1] = (0.0 <= reError <= max_reError)
            stop_conditions[2] = (0 <= rounds <= m)
            stop_conditions[3] = (0.0 <= seconds <= time.time() - current_time)

        # Determine stop condition (if it is added)
        for i, condition in enumerate(stop_conditions):
            if condition: 
                stop = actions[i]

        # Check reError
        if (interval[0] < 0 and interval[1] >= 0) or reError < 0:
            reError = None
        
        # Check abError
        if abError < 0: # Error value
            reError = None
            abError = None
            interval = [None, None]

        return u, interval, abError, reError, m, stop

    def estimationDistribution(self, stop_conditions, nr_of_steps):
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]

        # confidence level
        c = self._cPointEstimate(confidence)

        # global list counting final states of each trace
        self.state_count = [0] * len(self._states)
        distribution = [0] * len(self._states)
        interval = [[0, 0]] * len(self._states)
        abError = [-1.0] * len(self._states)
        reError = [-1.0] * len(self._states)
        m = 0

        # There are in total four applicable stop conditions for this function
        actions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        stop_conditions = [False] * 4

        # Save current time
        current_time = time.time()
        while not any(stop_conditions):

            self._markovSimulation([
                [lambda n, state: n, lambda c : 0 <= nr_of_steps <= c], # Exit when n is number of steps
                [lambda n, state: time.time() - current_time, lambda c : 0 <= seconds <= c], # Exit on time
                [lambda n, state: self._hittingStateCount(n, state, nr_of_steps), lambda c : False]
            ])

            # update m
            m += 1

            # calculate distributions:
            for i in range(len(distribution)):
                distribution[i] = self.state_count[i] / m
                nr_of_zeros = m - self.state_count[i]     
                Sm = np.sqrt((1/m)*( self.state_count[i]*pow(1 - distribution[i],2) + nr_of_zeros*pow(0 - distribution[i],2) ))

                # absolute error calculation
                abError[i] = abs((c*Sm) / np.sqrt(m))
                d = distribution[i]-abError[i]
                if d != 0.0:
                    reError[i] = abs(abError[i]/d)

                # interval calculation
                interval[i] = [distribution[i] - abError[i], distribution[i] + abError[i]]

            # Law of strong numbers is reliable after _law steps
            if m < MarkovChain._law and m != rounds: # (if rounds is less than 10 we still want an abError and reError)
                abError = [-1.0] * len(self._states)
                reError = [-1.0] * len(self._states)

            # Check stop conditions
            stop_conditions[0] = (0 <= max(abError) <= max_abError)
            stop_conditions[1] = (0 <= max(reError) <= max_reError)
            stop_conditions[2] = (0 <= rounds <= m)
            stop_conditions[3] = (0.0 <= seconds <= time.time() - current_time)

        # Determine stop condition (if it is added)
        for i, condition in enumerate(stop_conditions):
            if condition: 
                stop = actions[i]

        # maximum of all absolute and relative errors
        max_abError = max(abError)
        max_reError = max(reError)

        if not(sum(distribution) == 1.0):
            distribution = None

        if max_reError < 0 or any(i[0] < 0 and i[1] >= 0 for i in interval):
            max_reError = None
        
        if max_abError == -1.0:
            max_abError = None
            interval = None
            
        return distribution, interval, max_abError, max_reError, m, stop

    def estimationHittingState(self, stop_conditions, hitting_state, mode, func, states):
        # mode is used to check the hitting probability or the cumulative reward
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        nr_of_steps = stop_conditions[3]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]

        # confidence level
        c = self._cPointEstimate(confidence)

        # storing confidence interval information
        intervals = []
        abErrors = []
        reErrors = []
        m_list = []
        u_list = []

        # Calculate the hitting probability of single state or set of states
        if mode:
            # For single state hitting (we skip first state in trace simulation)
            compare = 0
        else:
            # For set of state hitting (we take first state into account)
            compare = -1

        # There are in total four applicable stop conditions for this function
        actions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        stop_conditions = [False] * 4
        stop = []

        # Save current time
        current_time = time.time()
        for initial_state in states:
            
            # manipulate the initial probabilities to start the markov chain in the required state i
            for state in self.states():
                if state == initial_state:
                    self._initialProbabilities[state] = 1.0
                else:
                    self._initialProbabilities[state] = 0.0

            nr_of_valid_traces = 0
            sum_hitting_rewards = 0.0
            M2 = 0.0
            
            interval = [0, 0]
            abError = -1.0
            reError = -1.0
            samples = []
            m = 0
            u = 0
            Sm = 0

            stop_conditions = [False] * 4

            while not any(stop_conditions):

                self.sum_rewards = 0.0
                self.valid_trace = False

                self._markovSimulation([
                    [lambda n, state : n, lambda c : 0 <= nr_of_steps <= c], # Exit when n is number of steps
                    [lambda n, state: time.time() - current_time, lambda c : 0 <= seconds <= c], # Exit on time
                    [lambda n, state : self._traceReward(n > compare, state, hitting_state), lambda c : c] # stop when hitting state is found
                ])
                
                m += 1
                if self.valid_trace:
                    nr_of_valid_traces += 1
                
                # ---- Calculate the confidence interval and ab/re errors --- #
                # - hitting probability confidence
                if func:
                    u = nr_of_valid_traces / m
                    nr_of_zeros = m - nr_of_valid_traces    
                    Sm = np.sqrt((1/m)*( nr_of_valid_traces*pow(1 - u,2) + nr_of_zeros*pow(0 - u,2) ))
                    abError = abs((c*Sm) / np.sqrt(m))
                
                # - Reward confidence
                else:
                    if self.valid_trace:
                        d1 = self.sum_rewards - u
                        u += d1/nr_of_valid_traces
                        d2 = self.sum_rewards - u
                        M2 += d1 * d2
                        Sm = np.sqrt(M2/nr_of_valid_traces)
                        abError = abs((c*Sm) / np.sqrt(nr_of_valid_traces))

                d = u-abError
                if d != 0.0:
                    reError = abs(abError/d)

                # interval calculation
                interval = [u - abError, u + abError]

                # Law of strong numbers is reliable after _law steps
                if m < MarkovChain._law and m != rounds: # (if rounds is less than 30 we still want an abError and reError)
                    abError = -1.0
                    reError = -1.0

                # Check stop conditions
                stop_conditions[0] = (0 <= abError <= max_abError)
                stop_conditions[1] = (0 <= reError <= max_reError)
                stop_conditions[2] = (0 <= rounds <= m)
                stop_conditions[3] = (0.0 <= seconds <= time.time() - current_time)

                # Determine stop condition (if it is added)
                for i, condition in enumerate(stop_conditions):
                    if condition: 
                        stop.append(actions[i])

            if reError < 0 or (interval[0] < 0 and interval [1] >= 0):
                reError = None
                
            if abError < 0:
                abError = None
                interval = [None, None]
                u = None
                

            m_list.append(m)
            u_list.append(u)
            abErrors.append(abError)
            reErrors.append(reError)
            intervals.append(interval)

            # Check if last stop condition is time
            if stop[-1] == actions[3]:
                break 

        if not func:
            for i in range(len(u_list)):
                if (u_list[i] == None) or (u_list[i] == 0.0):
                    u_list[i] = "Cannot be decided"

        return (
            u_list, 
            m_list,
            abErrors,
            reErrors,
            intervals,
            stop
        )
