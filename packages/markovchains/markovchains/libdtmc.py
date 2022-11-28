from fractions import Fraction
from io import StringIO
from typing import AbstractSet, Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

# pygraph library from:
# https://pypi.org/project/python-graph/
# https://github.com/Shoobx/python-graph

import pygraph.classes.digraph  as pyg
import pygraph.algorithms.accessibility as pyga
import pygraph.algorithms.searching as pygs
import functools
import math
import random
import os
import time
from statistics import NormalDist
from markovchains.libdtmcgrammar import parseDTMCDSL
import markovchains.utils.linalgebra as linalg
from markovchains.utils.utils import sortNames
from markovchains.utils.statistics import Statistics, DistributionStatistics

TStoppingCriteria = Tuple[float,float,float,int,int,float]

class DTMCException(Exception):
    pass
class MarkovChain(object):

    # states is a list of strings
    _states: List[str]
    # transitions maps states to another dictionary that maps target states to the probability of reaching that target state
    _transitions: Dict[str, Dict[str,Fraction]]
    # initialProbabilities maps states to the probability of starting in a given state
    _initialProbabilities: Dict[str,Fraction]
    # rewards is a dictionary that maps states to their rewards
    _rewards: Dict[str,Fraction]
    # transitionMatrix holds the transition matrix of the Markov Chains if it is computed
    _transitionMatrix: Optional[linalg.TMatrix]
    # initialProbabilityVector holds the initial probabilities in the form of a vector
    _initialProbabilityVector: Optional[linalg.TVector]
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
    
    def setInitialProbability(self, s: str, p: Fraction):
        '''Set initial probability of state s to p.'''
        if not s in self._states:
            raise DTMCException('Unknown state')
        self._initialProbabilities[s] = p

    def setReward(self, s: str, r: Fraction):
        '''Set reward of state s to r.'''
        if not s in self._states:
            raise DTMCException('Unknown state')
        self._rewards[s] = r

    def getReward(self, s: str)->Fraction:
        '''Get reward of state s. Defaults to 0 if undefined.'''
        if not s in self._rewards:
            return Fraction(0)
        return self._rewards[s]

    def setEdgeProbability(self, s: str, d: str, p: Fraction):
        '''Set the probability of the transition from s to d to p.'''
        if not s in self._states or not d in self._states:
            raise DTMCException('Unknown state')
        if s not in self._transitions:
            self._transitions[s] = dict()
        self._transitions[s][d] = p
    
    def transitions(self)->Set[Tuple[str,Fraction,str]]:
        '''Get the transitions of the dtmc as tuples (s, p, d). With source state s, destination state d and probability p.'''
        result = set()
        for i, srcstate in enumerate(self._transitions):
            for j, (dststate, p) in enumerate(self._transitions[srcstate].items()):
                result.add((srcstate, p, dststate))
        return result

    def addImplicitTransitions(self):
        '''Add the implicit transitions when the outgoing probabilities do not add up to one.'''
        # compute the transition matrix, which will have the implicit transition probabilities
        M: linalg.TMatrix = self.transitionMatrix()
        # set all transitions according to the non-zero elements in the matrix
        N = len(self._states)
        for i in range(N):
            si = self._states[i]
            for j in range(N):
                sj = self._states[j]
                if not M[j][i]==Fraction(0):
                    # add the transition if it does not yet exist
                    if not si in self._transitions:
                        self.setEdgeProbability(si, sj, M[j][i])
                    else:
                        if not sj in self._transitions[si]:
                            self.setEdgeProbability(si, sj, M[j][i])

    def _completeTransitionMatrix(self):
        '''Complete the transition matrix with missing/implicit transitions.'''
        if self._transitionMatrix is None:
            raise DTMCException("Transition matrix is not yet initialized.")
        # ensure that all rows add up to 1.
        # compute the row sums
        sums = linalg.rowSum(self._transitionMatrix)
        for n in range(len(self._states)):
            # if the row n sum is smaller than 1
            if sums[n] < Fraction(1):
                # try to add the missing probability mass on a self-loop on n, if it is not specified (is zero)
                if self._transitionMatrix[n][n] == Fraction(0):
                    self._transitionMatrix[n][n] = Fraction(1) - sums[n]
                else:
                    # cannot interpret it as an implicit transition
                    raise DTMCException("probabilities do not add to one")
            else:
                if sums[n] > Fraction(1):
                    raise DTMCException("probability mass is larger than one")

    def transitionMatrix(self)->linalg.TMatrix:
        '''Computes and returns the transition matrix of the MC.'''
        N = len(self._states)
        self._transitionMatrix = linalg.zeroMatrix(N, N)
        row = 0
        for ss in self._states:
            if ss in self._transitions:
                col = 0
                for sd in self._states:
                    if sd in self._transitions[ss]:
                        self._transitionMatrix[col][row] = self._transitions[ss][sd]
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
        sum = linalg.vectorSum(self._initialProbabilityVector)
        if sum > Fraction(1):
            raise DTMCException("probability is larger than one")
        if sum < Fraction(1):
            K = [self.initialProbabilitySpecified(s) for s in self._states].count(False)
            if K == 0:
                raise DTMCException("probability mass is smaller than one")
            remainder: Fraction = (Fraction(1) - sum) / Fraction(K)
            k = 0
            for s in self._states:
                if not self.initialProbabilitySpecified(s):
                    self._initialProbabilityVector[k] = remainder
                k += 1

    def completeInitialProbabilities(self):
        '''Complete the initial probabilities.'''
        # ensure that the initial probabilities add up to 1.
        sum = functools.reduce(lambda a,b : a+b, self._initialProbabilities.values(), Fraction(0))
        if sum > Fraction(1):
            raise DTMCException("initial probability mass is larger than one")
        if sum < Fraction(1):
            K = [self.initialProbabilitySpecified(s) for s in self._states].count(False)
            if K == 0:
                raise DTMCException("initial probability mass is smaller than one")
            remainder: Fraction = (Fraction(1) - sum) / Fraction(K)
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
                self.setReward(s, Fraction(0))

    def initialProbabilityVector(self)->linalg.TVector:
        '''Determine and return the initial probability vector.'''
        N = len(self._states)
        self._initialProbabilityVector = linalg.zeroVector(N)
        k = 0
        for s in self._states:
            if s in self._initialProbabilities:
                self._initialProbabilityVector[k] = self._initialProbabilities[s]
            k += 1
        self.completeInitialProbabilityVector()
        return self._initialProbabilityVector

    def rewardVector(self)->linalg.TVector:
        '''Return reward vector.'''
        return [self.getReward(s) for s in self._states]

    def rewardForDistribution(self, d: linalg.TVector)->Fraction:
        '''Return expected reward for a given distribution.'''
        result = Fraction(0)
        for k in range(self.numberOfStates()):
            result += d[k] * self.getReward(self._states[k])
        return result

    def executeSteps(self, N: int)->List[linalg.TVector]:
        '''Perform N steps of the chain, return array of N+1 distributions, starting with the initial distribution and distributions after N steps.'''
        if N<0:
            raise(DTMCException("Number of steps must be non-negative."))
        P = self.transitionMatrix()
        pi = self.initialProbabilityVector()
        result = []
        for _ in range(N+1):
            result.append(pi)
            pi = linalg.vectorMatrixProduct(pi, P)
        return result


    def _computeCommunicationGraph(self)->pyg.digraph:
        '''Return the communication graph corresponding to the Markov Chain.'''
        gr = pyg.digraph()
        gr.add_nodes(self._states)
        for s in self._transitions:
            for t in self._transitions[s]:
                if not self._transitions[s][t] == Fraction(0):
                    gr.add_edge((s, t))
        return gr

    def _computeReverseCommunicationGraph(self)->pyg.digraph:
        '''Return the communication graph corresponding to the reversed transitions of the Markov Chain.'''
        gr = pyg.digraph()
        gr.add_nodes(self._states)
        for s in self._transitions:
            for t in self._transitions[s]:
                if not self._transitions[s][t]==Fraction(0):
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

    def _hittingProbabilities(self, targetState: str)->Tuple[List[str],Optional[linalg.TMatrix],Dict[str,int],Dict[str,int],Dict[str,Fraction]]:
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
        solX: Optional[linalg.TVector] = None

        if  len(rs) > 0:

            # determine the hitting prob equations
            # Let us fix a state j ∈ S and define for each state i in rs a corresponding variable x_i (representing the hit probability f_ij ). Consider the system of linear equations defined by
            # x_i = Pij + Sum _ k∈S\{j} P_ik x_k
            # solve the equation: x = EQ x + Pj
            # solve the equation: (I-EQ) x = Pj

            N = len(rs)
            # initialize matrix I-EQ from the equation, and vector Pj
            ImEQ = linalg.identityMatrix(N)
            pj = linalg.zeroVector(N)
            # for all equations (rows of the matrix)
            for i in range(N):
                ip = pIndex[rs[i]]
                pj[i] = P[jp][ip]
                # for all variables in the summand
                for k in range(N):
                    # note that the sum excludes the target state!
                    if rs[k] != targetState:
                        kp = pIndex[rs[k]]
                        # determine row i, column k
                        ImEQ[k][i] -= P[kp][ip]

            # solve the equation x = inv(I-EQ)*Pj
            solX = linalg.solve(ImEQ, pj)

        # set all hitting probabilities to zero
        res: Dict[str,Fraction] = dict()
        for s in self._states:
            res[s] = Fraction(0)

        # fill the solutions from the equation
        for s in rs:
            solXv: linalg.TVector = solX  # type: ignore we know solX is a vector
            res[s] = solXv[rIndex[s]]

        return rs, ImEQ, rIndex, pIndex, res


    def hittingProbabilities(self, targetState: str)->Dict[str,Fraction]:
        '''Determine the hitting probabilities to hit targetState.'''

        _, _, _, _, res = self._hittingProbabilities(targetState)
        return res

    def rewardTillHit(self, targetState: str)->Dict[str,Fraction]:
        '''Determine the expected reward until hitting targetState'''

        rs, ImEQ, rIndex, pIndex, f = self._hittingProbabilities(targetState)
        solX = None
        if  len(rs) > 0:

            # x_i = r(i) · fij + Sum k∈S\{j} P_ik x_k
            # solve the equation: x = EQ x + Fj
            # solve the equation: (I-EQ) x = Fj

            N = len(rs)
            fj = linalg.zeroVector(N)
            for i in range(N):
                si = rs[i]
                fj[i] = self.getReward(si) * f[si]

            # solve the equation x = inv(I-EQ)*Fj
            ImEQm: linalg.TMatrix = ImEQ  # type: ignore we know ImEQ is a matrix
            solX = linalg.solve(ImEQm, fj)
            
        # set fr to zero
        fr: Dict[str,Fraction] = dict()
        for s in self._states:
            fr[s] = Fraction(0)

        # fill the solutions from the equation
        for s in rs:
            solXv: linalg.TVector = solX  # type: ignore we know solX is a vector
            fr[s] = solXv[rIndex[s]]

        res = dict()
        for s in rs:
            res[s] = fr[s] / f[s]

        return res


    def _hittingProbabilitiesSet(self, targetStates: List[str])->Tuple[linalg.TMatrix,List[str],Optional[linalg.TMatrix],Dict[str,int],Dict[str,int],Dict[str,Fraction]]:
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
        ImEQ = linalg.identityMatrix(N)
        sp = linalg.zeroVector(N)
        for i in range(N):
            ip = pIndex[rs[i]]
            # compute the i-th element in vector SP
            for s in targetStates:
                sp[i] += P[pIndex[s]][ip]

            # compute the i-th row in matrix ImEQ
            for k in range(N):
                kp = pIndex[rs[k]]
                # determine row i, column k
                ImEQ[k][i] -= P[kp][ip]

        # solve the equation x = inv(I-EQ)*SP
        solX = linalg.solve(ImEQ, sp)

        # set all hitting probabilities to zero
        res: Dict[str,Fraction] = dict()
        for s in self._states:
            res[s] = Fraction(0)

        # set all hitting probabilities in the target set to 1
        for s in targetStates:
            res[s] = Fraction(1)

        # fill the solutions from the equation
        for s in rs:
            res[s] = solX[rIndex[s]]

        return P, rs, ImEQ, rIndex, pIndex, res

    def hittingProbabilitiesSet(self, targetStates: List[str])->Dict[str,Fraction]:
        '''Determine the hitting probabilities to hit a set targetStates.'''
        _, _, _, _, _, res = self._hittingProbabilitiesSet(targetStates)
        return res

    def rewardTillHitSet(self, targetStates: List[str]):
        '''Determine the expected reward until hitting set targetStates.'''

        _, rs, ImEQ, rIndex, _, h = self._hittingProbabilitiesSet(targetStates)

        solX = None

        if  len(rs) > 0:

            # x_i = sum _ k in rs P_ik x_k  +  hh(i) = r(i) · h_i    for all i in rs

            # solve the equation: x = EQ x + H
            # solve the equation: (I-EQ) x = H

            N = len(rs)
            hh = linalg.zeroVector(N)
            for i in range(N):
                # compute the i-th element in vector H
                si = rs[i]
                hh[i] = self.getReward(si) * h[si]

            # solve the equation x = inv(I-EQ)*H
            ImEQm: linalg.TMatrix = ImEQ  # type: ignore we know that ImEQ is a matrix
            solX = linalg.solve(ImEQm, hh)

        # set hr to zero
        hr: Dict[str,Fraction] = dict()
        for s in self._states:
            hr[s] = Fraction(0)

        # fill the solutions from the equation
        for s in rs:
            solXv: linalg.TVector = solX  # type: ignore we know that solX is a vector
            hr[s] = solXv[rIndex[s]]

        res: Dict[str,Fraction] = dict()
        for s in targetStates:
            res[s] = Fraction(0)
        for s in rs:
            res[s] = hr[s] / h[s]

        return res

    def _getSubTransitionMatrixIndices(self, indices: List[int])->linalg.TMatrix:
        '''Return sub transition matrix consisting of the given list of indices.'''
        if self._transitionMatrix is None:
            raise DTMCException("Transition matrix has not been determined.")
        N = len(indices)
        res = linalg.zeroMatrix(N, N)
        for k in range(N):
            for m in range(N):
                res[k][m] = self._transitionMatrix[indices[k]][indices[m]]
        return res

    def _getSubTransitionMatrixClass(self, C: AbstractSet[str])->Tuple[Dict[str,int],linalg.TMatrix]:
        '''Return an index for the states in C and a sub transition matrix for the class C.'''
        # get submatrix for a class C of states
        indices = sorted([self._states.index(s) for s in C])
        index = dict([(c, indices.index(self._states.index(c))) for c in C])
        return index, self._getSubTransitionMatrixIndices(indices)

    def limitingMatrix(self)->linalg.TMatrix:
        '''Determine the limiting matrix of the dtmc.'''
        # formulate and solve balance equations for each of the  recurrent classes
        # determine the recurrent classes
        self.transitionMatrix()

        N = len(self._states)
        res = linalg.zeroMatrix(N, N)

        _, rClasses =  self.classifyTransientRecurrentClasses()

        # for each recurrent class:
        for c in rClasses:
            index, Pc = self._getSubTransitionMatrixClass(c)
            # a) solve the balance equations, pi P = pi I , pi.1 = 1
            #       pi (P-I); 1 = [0 1],
            M = len(c)
            PmI = linalg.subtractMatrix(Pc, linalg.identityMatrix(M))
            Q = linalg.addMatrix(linalg.matrixMatrixProduct(PmI, linalg.transpose(PmI)), linalg.oneMatrix(M,M))

            QInv = linalg.invertMatrix(Q)
            pi = linalg.columnSum(QInv)
            h = self.hittingProbabilitiesSet(list(c))
            # P(i,j) = h_i * pi j
            for sj in c:
                j = self._states.index(sj)
                for i in range(N):
                    if self._states[i] in c:
                        res[j][i] = pi[index[sj]]
                    else:
                        res[j][i] = h[self._states[i]] * pi[index[sj]]
        return res

    def limitingDistribution(self)->linalg.TVector:
        '''Determine the limiting distribution.'''
        P = self.limitingMatrix()
        pi0 = self.initialProbabilityVector()
        return linalg.vectorMatrixProduct(pi0, P)

    def longRunReward(self)-> Fraction:
        '''Determine the long-run expected reward.'''
        pi = self.limitingDistribution()
        r = self.rewardVector()
        return linalg.innerProduct(pi, r)

    @staticmethod
    def fromDSL(dslString: str)->Tuple[Optional[str],Optional['MarkovChain']]:

        factory = dict()
        factory['Init'] = lambda : MarkovChain()
        factory['AddState'] = lambda dtmc, s: (dtmc.addState(s), s)[1]
        factory['SetInitialProbability'] = lambda dtmc, s, p: dtmc.setInitialProbability(s, p)
        factory['SetReward'] = lambda dtmc, s, r: dtmc.setReward(s, r)
        factory['SetEdgeProbability'] = lambda dtmc, s, d, p: dtmc.setEdgeProbability(s, d, p)
        factory['SortNames'] = lambda dtmc: dtmc.sortStateNames()
        return parseDTMCDSL(dslString, factory)

    def __str__(self)->str:
        return str(self._states)


    # ---------------------------------------------------------------------------- #
    # - Markov Chain simulation                                                   - #
    # ---------------------------------------------------------------------------- #

    def setSeed(self, seed: int):
        ''' Set random generator seed'''
        random.seed(seed)

    def randomInitialState(self)->str:
        '''Return random initial state according to initial state distribution''' 
        r = random.random()
        p: float = 0.0

        for s in self._initialProbabilities:
            p = p + self._initialProbabilities[s]
            if r < p:
                return s
        # probability 0 of falling through to this point
        return self._states[0]

    def randomTransition(self, s: str)->str:
        '''Determine random transition.'''
        # calculate random value for state transition
        r = random.random()
        # set probability count to zero
        p = 0.0
        # Look through all transition probabilities of current state.
        for t in self._transitions[s]:
                p = p + self._transitions[s][t]
                if r < p:
                    return t
        # probability 0 of falling through to this point
        return s


    def setRecurrentState(self, state:Optional[str]):
        '''Set the recurrent state for simulation. If the given state is not a recurrent state. It is set to None.'''
        if state is None:
            self._recurrent_state = None
        else:
            if self._isRecurrentState(state):
                self._recurrent_state = state
            else:
                self._recurrent_state = None


    TSimulationAction = Callable[[int,str],bool]
    
    def _markovSimulation(self, actions: List[Tuple[TSimulationAction,Optional[str]]])->Tuple[int,Optional[str]]:
        '''Simulate Markov Chain. 
        actions is a list of pairs consisting of a callable that is called upon every step of the simulation and an optional string that describes the reason why the simulation ends. 
        The callable should take two arguments: n: int, the number of performed simulation steps before this one, and state: str, the current state of the Markov Chain in the simulation. It should return a Boolean value indicating if the simulation should be ended.
        Returns a pair n, stop, consisting of the total number of steps simulated and the optional string describing the reason for stopping.
        '''
        # Set current state to None to indicate that the first state is not defined
        current_state: Optional[str] = None
        stop = None

        # list for stop condition status
        stop_conditions:List[bool] = [False] * len(actions)

        # Calculate random markov chain sequence
        n: int = 0

        # Determine the random initial state
        current_state = self.randomInitialState()

        while not any(stop_conditions):            
            # perform simulation actions
            for i, (action,_) in enumerate(actions):
                stop_conditions[i] = action(n, current_state)

            # next random step in Markov Chain simulation
            current_state = self.randomTransition(current_state)
            n += 1

        # Determine stop condition
        for i, (_,st) in enumerate(actions):
            if stop_conditions[i]:
                stop = st

        return n, stop

    def _isRecurrentState(self, state: str)->bool:
        '''Check if state is a recurrent state.'''
        # recurrent state thats encountered in the random state_sequence dictionary
        _, rClasses = self.classifyTransientRecurrent()
        return state in rClasses

    def _cycleUpdate(self, state: str)->int:
        '''Called during simulation. Update statistics. Returns current cycle count.'''
        # Find first recurrent state
        if self._recurrent_state is None:
            if self._isRecurrentState(state):
                self._recurrent_state = state
            else:
                return -1 # To prevent the simulation from exiting
        
        if self._recurrent_state == state:
            self._statistics.visitRecurrentState()

        self._statistics.addReward(float(self._rewards[state]))
        return self._statistics.cycleCount()

    def _cycleUpdateCezaro(self, state: str)->int:
        '''Called during simulation. Update statistics. Returns current cycle count.'''

        # Find first recurrent state
        if self._recurrent_state is None:
            if self._isRecurrentState(state):
                self._recurrent_state = state
            else:
                return -1 # To prevent the simulation from exiting

        if self._recurrent_state == state:
            self._distributionStatistics.visitRecurrentState()

        self._distributionStatistics.addReward(self._states.index(state))

        return self._distributionStatistics.cycleCount()

    def _lastStateReward(self, nr_of_steps: int, n: int, state: str):
        '''Record the last state reward.'''
        if n == nr_of_steps:
            # TODO: what is self.y? Where is it defined? Line 1233.
            self.y = float(self._rewards[state])

    def _hittingStateCount(self, n: int, state: str, nr_of_steps: int):
        '''Update hitting state count if we are at step nr_of_steps.'''
        if n == nr_of_steps:
            for i in range(len(self.state_count)):
                if state == self._states[i]:
                    self.state_count[i] += 1

    def _traceReward(self, condition: bool, state: str, goal_state: Union[str,Iterable[str]])->bool:
        '''Accumulate reward. If condition is False, return False, if condition is True return if goal state is reached.
        Returns if goal state is reached.'''
        # check state for goal state list or single goal state
        valid = state in goal_state if isinstance(goal_state, list) else state == goal_state

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

    def markovTrace(self, rounds: int)->List[str]:
        '''Simulate Markov Chain for the given number of rounds (steps).
        Returns a list of length rounds+1 of the states visited in the simulation. '''
        # Declare empty list for states
        state_list: List[str] = []

        def _collect_List(_: int, state: str)->bool:
            state_list.append(state)
            return False

        self._markovSimulation([
            (lambda n, _: n >= rounds, None), # Exit when n is number of rounds
            (_collect_List, None) # add state to list
        ])

        return state_list

    def longRunExpectedAverageReward(self, stop_conditions:TStoppingCriteria)->Tuple[Statistics, Optional[str]]:
        '''Estimate the long run expected average reward by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds
        
        Returns a tuple: each of the results can be None if they could not be determined.
        - Statistics with:
            - confidence interval
            - estimate of the absolute error
            - estimate of the relative error
            - point estimate of the mean
            - point estimate of the variance
        - the stop criterion applied as a string
        '''
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        max_path_length = stop_conditions[3]
        nr_of_cycles = stop_conditions[4]
        seconds = stop_conditions[5]

        # Global variables used during simulation
        self._statistics = Statistics(confidence)
        # TODO: remove the following variables
        self.l:None  # Current cycle length
        self.r: None  # Current cycle cumulative reward
        self.Em: None  # Cycle count (-1 to subtract unfinished cycle beforehand)
        self.El: None # Sum of cycle lengths
        self.Er: None  # Sum of cumulative rewards
        self.El2: None  # Sum of cycle length squared
        self.Er2: None  # Sum of cycle cumulative reward squared
        self.Elr: None  # Sum of cycle product length and cycle
        self.u: None  # Estimated mean
        self.Sm: None  # Estimated variance

        def _action_pointEstU(n:int, state:str)->bool:
            self._statistics.pointEstU()
            return False

        def _action_pointEstSm(n:int, state:str)->bool:
            self._statistics.pointEstSm()
            return False

        def _action_AbsErr(n:int, state:str)->bool:
            c = self._statistics.abError(n)
            return 0 <= c <= max_abError
            
        def _action_RelErr(n:int, state:str)->bool:
            c = self._statistics.reError(n)
            return 0 <= c <= max_reError

        def _action_CycleUpdate(n:int, state:str)->bool:
            c = self._cycleUpdate(state)
            return 0 <= nr_of_cycles <= c

        # Save current time
        current_time = time.time()
        _, stop = self._markovSimulation([
            (lambda n, _: 0 <= max_path_length <= n, "Maximum path length"), # Run until max path length has been reached
            (lambda _n, _state: 0 <= seconds <= time.time() - current_time, "Timeout"), # Exit on time
            (_action_CycleUpdate, "Number of cycles"), # find first recurrent state
            (_action_pointEstU,None), # update point estimate of u
            (_action_pointEstSm,None), # update point estimate of Sm
            (_action_AbsErr, "Absolute Error"), # update point estimate of Sm
            (_action_RelErr, "Relative Error")
        ])

        # Compute absolute/relative error regardless of law of strong numbers
        abError = self._statistics.abError(MarkovChain._law)
        reError = self._statistics.reError(MarkovChain._law)

        # Compute confidence interval
        interval = self._statistics.confidenceInterval()

        # TODO: move the following into statistics?
        # Check reError
        if (interval[0] < 0 and interval[1] >= 0) or reError < 0:
            reError = None
        
        # Check abError
        if abError < 0: # Error value
            reError = None
            abError = None
            interval = None

        return self._statistics, stop
        # return interval, abError, reError, self.u, self.Em, stop

    def cezaroLimitDistribution(self, stop_conditions:TStoppingCriteria)-> Tuple[DistributionStatistics, Optional[str]]:
        '''
        Estimate the Cesaro limit distribution by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds
        
        Returns a tuple: each of the results can be None if they could not be determined.
        - DistributionStatistics with:
            - List of point estimates of the probabilities of the limit distribution
            - List of confidence intervals
            - List of estimates of the absolute errors
            - List of estimates of the relative errors
            - number of cycles 
        - the stop criterion applied as a string
        '''

        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        max_path_length = stop_conditions[3]
        nr_of_cycles = stop_conditions[4]
        seconds = stop_conditions[5]

        # Global variables
        # TODO: remove the following variables:
        self.Dist_l: None
        self.Dist_Em: None
        self.Dist_El: None
        self.Dist_El2: None
        self.Dist_rl: None
        self.Dist_Er: None
        self.Dist_Er2: None
        self.Dist_Elr: None
        self.Dist_u: None
        self.Dist_Sm: None

        self._distributionStatistics = DistributionStatistics(self.numberOfStates(), confidence)
        # self.Dist_l: int = 0 # Current cycle length
        # self.Dist_Em: int = -1 # Cycle count (-1 to subtract unfinished cycle beforehand)
        # self.Dist_El: int = 0 # Sum of cycle lengths
        # self.Dist_El2: int = 0 # Sum of cycle length squared
        # self.Dist_rl: List[float] = [0.0] * nr_of_states # Current cycle cumulative reward
        # self.Dist_Er: List[float] = [0.0] * nr_of_states # Sum of cumulative rewards
        # self.Dist_Er2: List[float] = [0.0] * nr_of_states # Sum of cycle cumulative reward squared
        # self.Dist_Elr: List[float] = [0.0] * nr_of_states # Sum of cycle product length and cycle
        # self.Dist_u: List[float] = [0.0] * nr_of_states # Estimated mean
        # self.Dist_Sm: List[float] = [0.0] * nr_of_states # Estimated variance

        state_list = []

        def _action_number_of_cycles(n: int, state:str)->bool:
            c = self._cycleUpdateCezaro(state)
            return 0 <= nr_of_cycles <= c

        def _action_pointEstUCezaro(n:int, state:str)->bool:
            self._distributionStatistics.pointEstUCezaro()
            return False

        def _action_pointEstSmCezaro(n:int, state:str)->bool:
            self._distributionStatistics.pointEstSmCezaro()
            return False

        def _action_appendState(n:int, state:str)->bool:
            state_list.append(state)
            return False

        def _action_abErrorCezaro(n: int, state:str)->bool:
            c = self._distributionStatistics.abErrorCezaro(n)
            return 0 <= max(c) <= max_abError

        def _action_reErrorCezaro(n: int, state:str)->bool:
            c = self._distributionStatistics.reErrorCezaro(n)
            return 0 <= max(c) <= max_reError

        # Save current time
        current_time = time.time()
        _, stop = self._markovSimulation([
            (lambda n, state: 0 <= max_path_length <= n, "Maximum path length"), # Run until max path length has been reached
            (lambda n, state: 0 <= seconds <= time.time() - current_time, "Timeout"), # Exit on time
            (_action_number_of_cycles, "Number of cycles"), # find first recurrent state
            (_action_pointEstUCezaro, None), # update point estimate of u
            (_action_pointEstSmCezaro, None), # update point estimate of Sm
            (_action_abErrorCezaro, "Absolute Error"), # Calculate smallest absolute error
            (_action_reErrorCezaro, "Relative Error"), # Calculate smallest relative error
            (_action_appendState, None) # Calculate smallest relative error
        ])

        abErrorVal: List[float] = self._distributionStatistics.abErrorCezaro(MarkovChain._law)
        abError: List[Optional[float]] = [v for v in abErrorVal]

        reErrorVal: List[float] = self._distributionStatistics.reErrorCezaro(MarkovChain._law)
        reError: List[Optional[float]] = [v for v in reErrorVal]
        

        # TODO: move the following into statistics?

        intervals = self._distributionStatistics.confidenceIntervals()
        for i in range(self.numberOfStates()):

            iv = intervals[i]

            # Check reError
            if (iv[0] < 0 and iv[1] >= 0) or reErrorVal[i] < 0:
                reError[i] = None
            
            # Check abError
            if abErrorVal[i] < 0: # Error value
                reError[i] = None
                abError[i] = None
                intervals[i] = None

        # Check if sum of distribution equals 1 with .4 float accuracy
        pntEst: Optional[List[float]] = self.Dist_u
        if not 0.9999 < sum(self._distributionStatistics.pointEstimates()) < 1.0001:
            pntEst = None

        return self._distributionStatistics, stop
        return pntEst, intervals, abError, reError, self.Dist_Em, stop

    def estimationExpectedReward(self, stop_conditions:TStoppingCriteria, nr_of_steps)->Tuple[
        Optional[float],
        Optional[Tuple[float,float]],
        Optional[float],
        Optional[float],
        int,
        Optional[str]]:
        '''
        Estimate the expected reward by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds
        
        Returns a tuple: each of the results can be None if they could not be determined.
        - point estimate of the expected reward
        - confidence interval
        - estimate of the absolute error
        - estimate of the relative error
        - number of rounds 
        - the stop criterion applied as a string
        '''
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]

        # confidence level
        c = self._cPointEstimate(confidence)

        interval: Tuple[float,float] = (0, 0)
        abErrorVal: float = -1.0
        reErrorVal: float = -1.0
        u: float = 0.0
        m: int = 0
        M2: float = 0.0 # Welford's algorithm variable

        # There are in total four applicable stop conditions for this function
        stopDescriptions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        sim_stop_conditions: List[bool] = [False] * 4

        # define action to be performed during simulation
        def _action_lastStateReward(n: int, state: str)->bool:
            self._lastStateReward(nr_of_steps, n, state)
            return False

        # Save current time
        current_time = time.time()
        while not any(sim_stop_conditions):
            # Used in self_lastStateReward function to assign reward to y when hitting last state
            self.y: float = 0.0

            self._markovSimulation([
                (lambda n, state: 0 <= nr_of_steps <= n, None), # Exit when n is number of rounds
                (lambda n, state: 0 <= seconds <= time.time() - current_time, None), # Exit on time
                (_action_lastStateReward, None)
            ])

            # Execute Welford's algorithm to compute running standard derivation and mean
            m += 1
            d1 = self.y - u
            u += d1/m
            d2 = self.y - u
            M2 += d1 * d2
            Sm = math.sqrt(M2/float(m))

            # Compute absolute and relative errors
            abErrorVal = abs((c*Sm) / math.sqrt(float(m)))
            d = u-abErrorVal
            if d != 0.0:
                reErrorVal = abs(abErrorVal/d)

            # interval calculation
            interval = (u - abErrorVal, u + abErrorVal)

            # Do not evaluate abError/reError in the first _law cycles:
            if m < MarkovChain._law and m != rounds: # (if rounds is less than 10 we still want an abError and reError)
                abErrorVal = -1.0
                reErrorVal = -1.0

            # Check stop conditions
            sim_stop_conditions[0] = (0.0 <= abErrorVal <= max_abError)
            sim_stop_conditions[1] = (0.0 <= reErrorVal <= max_reError)
            sim_stop_conditions[2] = (0 <= rounds <= m)
            sim_stop_conditions[3] = (0.0 <= seconds <= time.time() - current_time)

        # Determine stop condition (if it is added)
        stop = None
        for i, condition in enumerate(sim_stop_conditions):
            if condition: 
                stop = stopDescriptions[i]

        # Check reError
        reError = reErrorVal
        if (interval[0] < 0 and interval[1] >= 0) or reErrorVal < 0:
            reError = None
        
        # Check abError
        abError = abErrorVal
        intervalResult = interval
        if abErrorVal < 0: # Error value
            reError = None
            abError = None
            intervalResult = None

        return u, intervalResult, abError, reError, m, stop


    def estimationDistribution(self, stop_conditions:TStoppingCriteria, nr_of_steps: int)->Tuple[
        Optional[List[float]],
        Optional[List[Tuple[float,float]]],
        Optional[float],
        Optional[float],
        int,
        Optional[str]]:
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]
        '''
        Estimate the distribution after nr_of_steps by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        -absolute error
        - relative error
        - path length
        - nr. rounds
        - timeout in seconds
        
        Returns a tuple: each of the results can be None if they could not be determined.
        - List of point estimates of the probabilities of the distribution
        - List of confidence intervals
        - max of estimates of the absolute errors
        - max of estimates of the relative errors
        - number of cycles 
        - the stop criterion applied as a string
        '''

        # confidence level
        c = self._cPointEstimate(confidence)

        # global list counting final states of each trace
        self.state_count = [0] * len(self._states)
        distribution: List[float] = [0.0] * len(self._states)
        interval: List[Tuple[float,float]] = [(0.0,0.0)] * len(self._states)
        abError = [-1.0] * len(self._states)
        reError = [-1.0] * len(self._states)
        m = 0

        # There are in total four applicable stop conditions for this function
        stopDescriptions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        sim_stop_conditions: List[bool] = [False] * 4

        # Save current time
        starting_time = time.time()
        
        # define action to be performed during simulation
        def _action_hittingStateCount(n: int, state: str)->bool:
            self._hittingStateCount(n, state, nr_of_steps)
            return False

        while not any(sim_stop_conditions):

            self._markovSimulation([
                (lambda n, state: 0 <= nr_of_steps <= n, None), # Exit when n is number of steps
                (lambda n, state: 0 <= seconds <= time.time() - starting_time, None), # Exit on time
                (_action_hittingStateCount, None)
            ])

            # update m
            m += 1

            # calculate distributions:
            for i in range(len(distribution)):
                distribution[i] = self.state_count[i] / m
                nr_of_zeros = m - self.state_count[i]     
                Sm = math.sqrt((1/float(m))*( self.state_count[i]*pow(1 - distribution[i],2) + nr_of_zeros*pow(0 - distribution[i],2) ))

                # absolute error calculation
                abError[i] = abs((c*Sm) / math.sqrt(float(m)))
                d = distribution[i]-abError[i]
                if d != 0.0:
                    reError[i] = abs(abError[i]/d)

                # interval calculation
                interval[i] = (distribution[i] - abError[i], distribution[i] + abError[i])

            # Law of strong numbers is reliable after _law steps
            if m < MarkovChain._law and m != rounds: # (if rounds is less than 10 we still want an abError and reError)
                abError = [-1.0] * len(self._states)
                reError = [-1.0] * len(self._states)

            # Check stop conditions
            sim_stop_conditions[0] = (0 <= max(abError) <= max_abError)
            sim_stop_conditions[1] = (0 <= max(reError) <= max_reError)
            sim_stop_conditions[2] = (0 <= rounds <= m)
            sim_stop_conditions[3] = (0.0 <= seconds <= time.time() - starting_time)

        # Determine stop condition (if it is added)
        stop = None
        for i, condition in enumerate(sim_stop_conditions):
            if condition: 
                stop = stopDescriptions[i]

        # maximum of all absolute and relative errors
        max_abError: float = max(abError)
        max_reError: float = max(reError)

        distributionResult = distribution
        if not(sum(distribution) == 1.0):
            distributionResult = None

        max_reErrorResult = max_reError
        if max_reError < 0 or any(i[0] < 0 and i[1] >= 0 for i in interval):
            max_reErrorResult = None
        
        intervalResult = interval
        max_abErrorResult = max_abError
        if max_abError == -1.0:
            max_abErrorResult = None
            intervalResult = None
            
        return distributionResult, intervalResult, max_abErrorResult, max_reErrorResult, m, stop


    def estimationHittingState(self, stop_conditions:TStoppingCriteria, hitting_state: Union[str,List[str]], mode: bool, func: bool, states: List[str])->Tuple[
        List[Optional[Union[float,str]]],
        List[int],
        List[Optional[float]],
        List[Optional[float]],
        List[Optional[Tuple[float,float]]],
        List[Optional[str]]]:
        '''
        Estimate the hitting probability or expected reward until hitting either a single state, or the state set by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        Parameter hitting_state is either a single state, or a set of states.
        Mode is a Boolean the indicates if we are using a single state (True) or a set of states (False). Note that the definition of hitting probability is different for both cases.
        Func is a Boolean that indicates if we determine hitting probability (True) or expected reward until hit (False).

        return (
            u_list, 
            m_list,
            abErrors,
            reErrors,
            intervals,
            stop
        )


        Returns a tuple: each of the results can be None if they could not be determined.
        - point estimates of the requested metric
        - confidence intervals
        - estimates of the absolute error
        - estimates of the relative error
        - number of rounds for ech state
        - the stop criteria applied as strings
        '''

        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        nr_of_steps = stop_conditions[3]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]

        # confidence level
        c = self._cPointEstimate(confidence)

        # storing confidence interval information
        intervalResults:List[Optional[Tuple[float,float]]] = []
        abErrorResults: List[Optional[float]] = []
        reErrorResults = []
        m_list:List[int] = []
        u_listResults: List[Optional[Union[float,str]]] = []

        # Calculate the hitting probability of single state or set of states
        if mode:
            # For single state hitting (we skip first state in trace simulation)
            compare = 0
        else:
            # For set of state hitting (we take first state into account)
            compare = -1

        # There are in total four applicable stop conditions for this function
        stopDescriptions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        sim_stop_conditions = [False] * 4
        stop: List[Optional[str]] = []

        # Save current time
        current_time = time.time()
        for initial_state in states:
            
            # manipulate the initial probabilities to start the markov chain in the required state i
            for state in self.states():
                if state == initial_state:
                    self._initialProbabilities[state] = Fraction(1)
                else:
                    self._initialProbabilities[state] = Fraction(0)

            nr_of_valid_traces = 0
            M2 = 0.0
            
            interval = (0.0, 0.0)
            abError = -1.0
            reError = -1.0
            m = 0
            u = 0
            Sm = 0

            sim_stop_conditions = [False] * 4

            # define action to be performed during simulation
            def _action_traceReward(n: int, state: str)->bool:
                return self._traceReward(n > compare, state, hitting_state)

            while not any(sim_stop_conditions):

                # TODO: these globals are a bit ugly, remove? group?
                self.sum_rewards: float = 0.0
                self.valid_trace: bool = False

                self._markovSimulation([
                    (lambda n, state: 0 <= nr_of_steps <= n, None), # Exit when n is number of steps
                    (lambda n, state: 0 <= seconds <= time.time() - current_time, None), # Exit on time
                    (_action_traceReward, None) # stop when hitting state is found
                ])
                
                m += 1
                if self.valid_trace:
                    nr_of_valid_traces += 1
                
                # ---- Calculate the confidence interval and ab/re errors --- #
                # - hitting probability confidence
                if func:
                    u = nr_of_valid_traces / m
                    nr_of_zeros = m - nr_of_valid_traces    
                    Sm = math.sqrt((1/float(m))*( nr_of_valid_traces*pow(1 - u,2) + nr_of_zeros*pow(0 - u,2) ))
                    abError = abs((c*Sm) / math.sqrt(float(m)))
                
                # - Reward confidence
                else:
                    if self.valid_trace:
                        d1 = self.sum_rewards - u
                        u += d1/nr_of_valid_traces
                        d2 = self.sum_rewards - u
                        M2 += d1 * d2
                        Sm = math.sqrt(M2/float(nr_of_valid_traces))
                        abError = abs((c*Sm) / math.sqrt(float(nr_of_valid_traces)))

                d = u-abError
                if d != 0.0:
                    reError = abs(abError/d)

                # interval calculation
                interval: Tuple[float,float] = (u - abError, u + abError)

                # Law of strong numbers is reliable after _law steps
                if m < MarkovChain._law and m != rounds: # (if rounds is less than 30 we still want an abError and reError)
                    abError = -1.0
                    reError = -1.0

                # Check stop conditions
                sim_stop_conditions[0] = (0 <= abError <= max_abError)
                sim_stop_conditions[1] = (0 <= reError <= max_reError)
                sim_stop_conditions[2] = (0 <= rounds <= m)
                sim_stop_conditions[3] = (0.0 <= seconds <= time.time() - current_time)

                # Determine stop condition (if it is added)
                for i, condition in enumerate(sim_stop_conditions):
                    if condition: 
                        stop.append(stopDescriptions[i])

            reErrorResult = reError
            if reError < 0 or (interval[0] < 0 and interval [1] >= 0):
                reErrorResult = None

            uResult: Optional[float] = u    
            abErrorResult = abError
            intervalResult: Optional[Tuple[float,float]]  = interval
            if abError < 0:
                abErrorResult = None
                intervalResult = None
                uResult = None
                

            m_list.append(m)
            u_listResults.append(uResult)
            abErrorResults.append(abErrorResult)
            reErrorResults.append(reErrorResult)
            intervalResults.append(intervalResult)

            # Check if last stop condition is time
            if stop[-1] == stopDescriptions[3]:
                break 

        if not func:
            for i in range(len(u_listResults)):
                if (u_listResults[i] == None) or (u_listResults[i] == 0.0):
                    u_listResults[i] = "Cannot be decided"

        return (
            u_listResults, 
            m_list,
            abErrorResults,
            reErrorResults,
            intervalResults,
            stop
        )
