from fractions import Fraction
from io import StringIO
from typing import AbstractSet, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

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
        for srcState in self._transitions:
            for (dstState, p) in self._transitions[srcState].items():
                result.add((srcState, p, dstState))
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
        # get sub-matrix for a class C of states
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
    # - Markov Chain simulation                                                  - #
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
        '''Determine random transition from state s.'''
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
        '''
        Set the recurrent state for simulation. If the given state is not a recurrent state, an exception is raised.
        If state is None, recurrent state is cleared.
        '''
        if state is None:
            self._recurrent_state = None
        else:
            if self._isRecurrentState(state):
                self._recurrent_state = state
            else:
                raise DTMCException("{} is not a recurrent state.".format(state))


    # type for generic actions to be executed during a random simulation
    TSimulationAction = Callable[[int,str],bool]
    
    def _markovSimulation(self, actions: List[Tuple[TSimulationAction,Optional[str]]], initialState: Optional[str] = None)->Tuple[int,Optional[str]]:
        '''
        Simulate Markov Chain. 
        actions is a list of pairs consisting of a callable that is called upon every step of the simulation and an optional string that describes the reason why the simulation ends. 
        The callable should take two arguments: n: int, the number of performed simulation steps before this one, and state: str, the current state of the Markov Chain in the simulation. It should return a Boolean value indicating if the simulation should be ended.
        An optional forced initial state can be provided. If no initial state is provided, it is selected randomly according to the initial state distribution.
        Returns a pair n, stop, consisting of the total number of steps simulated and the optional string describing the reason for stopping.
        '''
        # list for stop condition status
        stop_conditions:List[bool] = [False] * len(actions)

        # Step counter
        n: int = 0

        # Determine current state as random initial state
        if initialState is None:
            current_state: Optional[str] = self.randomInitialState()
        else:
            current_state = initialState

        while not any(stop_conditions):            
            # perform simulation actions
            for i, (action,_) in enumerate(actions):
                stop_conditions[i] = action(n, current_state)

            # next random step in Markov Chain simulation
            current_state = self.randomTransition(current_state)
            n += 1

        # Determine stop condition
        stop = None
        for i, (_,st) in enumerate(actions):
            if stop_conditions[i]:
                stop = st

        return n, stop

    def _isRecurrentState(self, state: str)->bool:
        '''Check if state is a recurrent state. Note that this method may be time consuming as an analysis is performed every time it is called.'''
        # recurrent state thats encountered in the random state_sequence dictionary
        _, rClasses = self.classifyTransientRecurrent()
        return state in rClasses

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

    def genericEstimationFromRecurrentCycles(self, stop_conditions:TStoppingCriteria, action_update: Callable[[Statistics,int,str],bool])->Tuple[Statistics, Optional[str]]:
        '''generic estimation framework by simulation using the provided stop_conditions.
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
        statistics = Statistics(confidence)

        recurrentState: Optional[str] = None
        recurrentStates: Set[str] = (self.classifyTransientRecurrent())[1]

        def _action_AbsErr(_n:int, _state:str)->bool:
            c = statistics.abError()
            if c is None:
                return False
            return max_abError > 0 and c <= max_abError
            
        def _action_RelErr(_n:int, _state:str)->bool:
            c = statistics.reError()
            if c is None:
                return False
            return max_reError > 0 and c <= max_reError
            
        def _action_CycleUpdate(n:int, state:str)->bool:
            nonlocal recurrentState, recurrentStates
            if recurrentState is None:
                if state in recurrentStates:
                    recurrentState = state
            if state != recurrentState:
                return False
            statistics.completeCycle()
            return 0 <= nr_of_cycles <= statistics.cycleCount()

        # TODO: suppress collection of data until first hit of recurrent state?
        def _action_update(n: int, state: str)->bool:
            return action_update(statistics, n, state)

        # Save current time
        current_time = time.time()
        _, stop = self._markovSimulation([
            (lambda n, _: 0 <= max_path_length <= n, "Maximum path length"), # Run until max path length has been reached
            (lambda _n, _state: 0 <= seconds <= time.time() - current_time, "Timeout"), # Exit on time
            (_action_update, None),
            (_action_CycleUpdate, "Number of cycles"), # check cycle completed
            (_action_AbsErr, "Absolute Error"), # check absolute error
            (_action_RelErr, "Relative Error") # check relative error
        ])
        return statistics, stop


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
        def _action_addSample(statistics: Statistics, _n:int, state:str)->bool:
            statistics.addSample(float(self._rewards[state]))
            return False

        return self.genericEstimationFromRecurrentCycles(stop_conditions, _action_addSample)


    def cezaroLimitDistribution(self, stop_conditions:TStoppingCriteria)-> Tuple[Optional[DistributionStatistics], Optional[str]]:
        '''
        Estimate the Cezaro limit distribution by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds
        
        Returns a tuple:
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

        distributionStatistics = DistributionStatistics(self.numberOfStates(), confidence)

        def _action_number_of_cycles(_n: int, _state:str)->bool:
            # TODO: is it correct tom complete cycle here? Next is not independent?
            # Shouldn't we end cycle on recurent state?
            # also: skip until first hit of recurrent state
            distributionStatistics.completeCycle()
            return 0 <= nr_of_cycles <= distributionStatistics.cycleCount()

        def _action_visitState(_n:int, state:str)->bool:
            distributionStatistics.visitState(self._states.index(state))
            return False

        def _action_abError(_n: int, _state:str)->bool:
            c = distributionStatistics.abError()
            if c is None:
                return False
            if any([v is None for v in c]):
                return False
            vc : List[float] = c  # type: ignore
            return max_abError > 0 and max(vc) <= max_abError

        def _action_reError(_n: int, _state:str)->bool:
            c = distributionStatistics.reError()
            if c is None:
                return False
            if any([v is None for v in c]):
                return False
            vc : List[float] = c  # type: ignore
            return max_reError > 0 and max(vc) <= max_reError

        # Save current time
        current_time = time.time()
        _, stop = self._markovSimulation([
            (lambda n, state: 0 <= max_path_length <= n, "Maximum path length"), # Run until max path length has been reached
            (lambda n, state: 0 <= seconds <= time.time() - current_time, "Timeout"), # Exit on time
            (_action_visitState, ""),
            (_action_number_of_cycles, "Number of cycles"), # find first recurrent state
            (_action_abError, "Absolute Error"), # Calculate smallest absolute error
            (_action_reError, "Relative Error") # Calculate smallest relative error
        ])
        
        return distributionStatistics, stop

    def estimationExpectedReward(self, stop_conditions:TStoppingCriteria, nr_of_steps)->Tuple[
        Statistics,
        Optional[str]]:
        '''
        Estimate the transient expected reward after nr_of_steps by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds
        
        Returns a tuple: each of the results can be None if they could not be determined.
        - statistics of the expected reward 
        - the stop criterion applied as a string
        '''
        # separate stop conditions
        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]

        stopDescriptions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]

        statistics = Statistics(confidence)

        def _action_lastStateReward(n: int, state: str)->bool:
            if n == nr_of_steps:
                statistics.addSample(float(self._rewards[state]))
                statistics.completeCycle()
            return False

        sim_stop_conditions: List[bool] = [False] * 4

        # Save current time
        current_time = time.time()
        while not any(sim_stop_conditions):
            self._markovSimulation([
                (_action_lastStateReward, None),
                (lambda n, state: 0 <= nr_of_steps <= n, None), # Exit when n is number of rounds
                (lambda n, state: 0 <= seconds <= time.time() - current_time, None), # Exit on time
            ])

            # Check stop conditions
            abErrorVal = statistics.abError()
            if abErrorVal is not None:
                sim_stop_conditions[0] = (0.0 <= abErrorVal <= max_abError)
            reErrorVal = statistics.reError()
            if reErrorVal is not None:
                sim_stop_conditions[1] = (0.0 <= reErrorVal <= max_reError)
            sim_stop_conditions[2] = (0 <= rounds <= statistics.cycleCount())
            sim_stop_conditions[3] = (0.0 <= seconds <= time.time() - current_time)

        # Determine stop condition (if it is added)
        stop = None
        for i, condition in enumerate(sim_stop_conditions):
            if condition: 
                stop = stopDescriptions[i]
       
        return statistics, stop


    def estimationTransientDistribution(self, stop_conditions:TStoppingCriteria, nr_of_steps: int)->Tuple[
        DistributionStatistics, Optional[str]]:
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
        
        Returns a tuple:
        - Statistics of the estimated distribution 
        - the stop criterion applied as a string
        '''

        distributionStatistics = DistributionStatistics(len(self._states), confidence)

        # There are in total four applicable stop conditions for this function
        stopDescriptions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        sim_stop_conditions: List[bool] = [False] * 4

        # Save current time
        starting_time = time.time()
        
        currentState: Optional[str] = None

        def _action_trackState(_n: int, state: str)-> bool:
            nonlocal currentState
            currentState = state
            return True
        
        while not any(sim_stop_conditions):

            self._markovSimulation([
                (_action_trackState, None),
                (lambda n, state: 0 <= nr_of_steps <= n, None), # Exit when n is number of steps
                (lambda n, state: 0 <= seconds <= time.time() - starting_time, None), # Exit on time out
            ])

            vCurrentState: str = currentState  # type: ignore
            distributionStatistics.visitState(self._states.index(vCurrentState))
            distributionStatistics.completeCycle()

            abError = distributionStatistics.abError()
            reError = distributionStatistics.reError()

            # Check stop conditions
            if not (None in abError):
                vAbError: List[float] = abError  # type: ignore
                sim_stop_conditions[0] = (0 <= max(vAbError) <= max_abError)
            if not (None in reError):
                vReError: List[float] = reError  # type: ignore
                sim_stop_conditions[1] = (0 <= max(vReError) <= max_reError)
            sim_stop_conditions[2] = (0 <= rounds <= distributionStatistics.cycleCount())
            sim_stop_conditions[3] = (0.0 <= seconds <= time.time() - starting_time)

        # Determine stop condition (if it is added)
        stop = None
        for i, condition in enumerate(sim_stop_conditions):
            if condition: 
                stop = stopDescriptions[i]
           
        return distributionStatistics, stop


    def estimationHittingStateGeneric(self, stop_conditions:TStoppingCriteria, analysisStates: List[str], initialization: Callable, action: Callable[[int,str],bool], onHit: Callable[[Statistics],None], onNoHit: Callable[[Statistics],None])->Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:
        
        '''
        Generic framework for estimating hitting probability, or reward until hit by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        The analysis is performed for all initial states in analysisStates

        initialization is called every time the analysis starts for a new starting state

        action is called every step of the simulation and should return a boolean indicating if the simulation should be stopped because the target set is hit

        after the simulation is finished, onHit is called if the target is hit (the maximum number of steps is not completed in the simulation).
        onNoHit is called if the target was not hit after the maximum number of steps.

        Returns a tuple:
        - statistics of the estimated hitting probability
        - the stop criteria applied as strings
        '''

        confidence = stop_conditions[0]
        max_abError = stop_conditions[1]
        max_reError = stop_conditions[2]
        nr_of_steps = stop_conditions[3]
        rounds = stop_conditions[4]
        seconds = stop_conditions[5]

        statistics: Dict[str,Statistics] = dict()
        for s in analysisStates:
            statistics[s] = Statistics(confidence)

        # There are in total four applicable stop conditions for this function
        stopDescriptions = ["Absolute Error", "Relative Error", "Number of Paths", "Timeout"]
        sim_stop_conditions = [False] * 4
        stop: Dict[str,str] = dict()

        # Save current time
        current_time = time.time()
        for initialState in analysisStates:
            
            sim_stop_conditions = [False] * 4

            # generic initialization
            initialization()

            while not any(sim_stop_conditions):

                _, simResult = self._markovSimulation([
                    (lambda n, state: 0 <= nr_of_steps <= n, "steps"), # Exit when n is number of steps
                    (lambda n, state: 0 <= seconds <= time.time() - current_time, "timeout"), # Exit on time
                    (action, "hit") # stop when hitting state is found
                ], initialState)

                if simResult=="timeout":
                   return None, "Timeout"

                if simResult!="steps":
                    # hitting state was hit
                    onHit(statistics[initialState])
                else:
                    # specific
                    onNoHit(statistics[initialState])

                # Check stop conditions
                abError = statistics[initialState].abError()
                if abError is not None:
                    sim_stop_conditions[0] = (0 <= abError <= max_abError)
                reError = statistics[initialState].reError()
                if reError is not None:
                    sim_stop_conditions[1] = (0 <= reError <= max_reError)
                sim_stop_conditions[2] = (0 <= rounds <= statistics[initialState].cycleCount())
                sim_stop_conditions[3] = (0.0 <= seconds <= time.time() - current_time)

            # Determine stop condition (if it is added)
            for i, condition in enumerate(sim_stop_conditions):
                if condition: 
                    stop[initialState] = stopDescriptions[i]
                
        return statistics, stop



    def estimationHittingProbabilityState(self, stop_conditions:TStoppingCriteria, hitting_state: str, analysisStates: List[str])->Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:
        
        '''
        Estimate the hitting probability until hitting a single state by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        Parameter hitting_state is a state to be hit

        The analysis is performed for all initial states in analysisStates
        
        Returns a tuple:
        - statistics of the estimated hitting probability
        - the stop criteria applied as strings
        '''

        def initialization():
            pass

        def action(n: int, state: str)->bool:
            # define action to be performed during simulation
            # suppress initial state for hitting
            if n == 0:
                return False
            return state == hitting_state

        def onHit(s: Statistics):
            s.addSample(1.0)
            s.completeCycle()

        def onNoHit(s: Statistics):
            s.addSample(0.0)
            s.completeCycle()

        return self.estimationHittingStateGeneric(stop_conditions, analysisStates, initialization, action, onHit, onNoHit)

    def estimationRewardUntilHittingState(self, stop_conditions:TStoppingCriteria, hitting_state: str, analysisStates: List[str])->Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:
        
        '''
        Estimate the cumulative reward until hitting a single state by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        Parameter hitting_state is a state to be hit

        The analysis is performed for all initial states in analysisStates
        
        Returns a tuple:
        - statistics of the cumulative reward
        - the stop criteria applied as strings
        '''

        accumulatedReward: float

        def initialization():
            nonlocal accumulatedReward
            accumulatedReward = 0.0

        def action(n: int, state: str)->bool:
            # define action to be performed during simulation
            nonlocal accumulatedReward
            if n==0:
                accumulatedReward += float(self.getReward(state))
                # suppress initial state for hitting
                return False
            if state == hitting_state:
                return True
            # reward of hitting state is npt counted
            accumulatedReward += float(self.getReward(state))
            return False

        def onHit(s: Statistics):
            s.addSample(accumulatedReward)
            s.completeCycle()

        def onNoHit(s: Statistics):
            pass

        return self.estimationHittingStateGeneric(stop_conditions, analysisStates, initialization, action, onHit, onNoHit)


    def estimationHittingProbabilityStateSet(self, stop_conditions:TStoppingCriteria, hitting_states: List[str], analysisStates: List[str])->Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:
        
        '''
        Estimate the hitting probability until hitting a set of states by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        Parameter hitting_states is a list of states to be hit

        The analysis is performed for all initial states in analysisStates
        
        Returns a tuple:
        - statistics of the estimated hitting probability
        - the stop criteria applied as strings
        '''

        def initialization():
            pass

        # define action to be performed during simulation
        def action(n: int, state: str)->bool:
            # suppress initial state for hitting
            if n == 0:
                return False
            return state in hitting_states

        def onHit(s: Statistics):
            s.addSample(1.0)
            s.completeCycle()

        def onNoHit(s: Statistics):
            s.addSample(0.0)
            s.completeCycle()

        return self.estimationHittingStateGeneric(stop_conditions, analysisStates, initialization, action, onHit, onNoHit)

    def estimationRewardUntilHittingStateSet(self, stop_conditions:TStoppingCriteria, hitting_states: List[str], analysisStates: List[str])->Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:
        
        '''
        Estimate the cumulative reward until hitting a single state by simulation using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        Parameter hitting_states is a list of states to be hit

        The analysis is performed for all initial states in analysisStates
        
        Returns a tuple:
        - statistics of the cumulative reward
        - the stop criteria applied as strings
        '''

        accumulatedReward: float

        def initialization():
            nonlocal accumulatedReward
            accumulatedReward = 0.0

        # define action to be performed during simulation
        def action(n: int, state: str)->bool:
            nonlocal accumulatedReward
            if n==0:
                accumulatedReward += float(self.getReward(state))
                # suppress initial state for hitting
                return False
            if state in hitting_states:
                return True
            # reward of hitting state is not counted
            accumulatedReward += float(self.getReward(state))
            return False

        def onHit(s: Statistics):
            s.addSample(accumulatedReward)
            s.completeCycle()

        def onNoHit(s: Statistics):
            pass

        return self.estimationHittingStateGeneric(stop_conditions, analysisStates, initialization, action, onHit, onNoHit)

