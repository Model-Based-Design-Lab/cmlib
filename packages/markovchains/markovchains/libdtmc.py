"""Library for Discrete-Time Markov Chain analysis."""

from fractions import Fraction
from io import StringIO
from typing import AbstractSet, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

# pygraph library from:
# https://pypi.org/project/python-graph/
# https://github.com/Shoobx/python-graph

import functools
import math
import random
import os
import pygraph.classes.digraph  as pyg
import pygraph.algorithms.accessibility as pyga
import pygraph.algorithms.searching as pygs
from markovchains.libdtmcgrammar import parse_dtmc_dsl
import markovchains.utils.linalgebra as linalg
from markovchains.utils.utils import sort_names, TimeoutTimer
from markovchains.utils.statistics import Statistics, DistributionStatistics, StopConditions

# type for generic actions to be executed during a random simulation
TSimulationAction = Callable[[int,str],bool]
TSimulationActionAndDescription = Tuple[TSimulationAction,Optional[str]]

STR_TIMEOUT = "Timeout"
STR_ABS_ERROR = "Absolute Error"
STR_REL_ERROR = "Relative Error"
STR_NR_OF_PATHS = "Number of Paths"
STR_MAX_PATH_LENGTH = "Maximum path length"

class DTMCException(Exception):
    """Exception from the DTMC package"""

class MarkovChain:
    """Representation of a Discrete-Tie Markov Chain"""

    # states is a list of strings
    _states: List[str]
    # transitions maps states to another dictionary that maps target states
    # to the probability of reaching that target state
    _transitions: Dict[str, Dict[str,Fraction]]
    # initialProbabilities maps states to the probability of starting in a given state
    _initial_probabilities: Dict[str,Fraction]
    # rewards is a dictionary that maps states to their rewards
    _rewards: Dict[str,Fraction]
    # transitionMatrix holds the transition matrix of the Markov Chains if it is computed
    _transition_matrix: Optional[linalg.TMatrix]
    # initialProbabilityVector holds the initial probabilities in the form of a vector
    _initial_probability_vector: Optional[linalg.TVector]
    # Recurrent state for simulation run calibration
    _recurrent_state: Optional[str]


    def __init__(self):
        self._states = []
        self._transitions = {}
        self._initial_probabilities = {}
        self._rewards = {}
        self._transition_matrix = None
        self._initial_probability_vector = None
        self._recurrent_state = None
        # Set pseudo random seed based on os current time
        random_data = os.urandom(8)
        seed = int.from_bytes(random_data, byteorder="big")
        random.seed(seed)

    def as_dsl(self, name: str, state_info:bool = True, reward_info:bool = True)->str:
        '''Return the model as a string of the domain-specific language.'''
        # keep track of the states that have been output
        states_output: Set[str] = set()
        # create string writer for the output
        output = StringIO()
        # write header
        output.write(f"markov chain {name} {{\n")

        for tr in sorted(self.transitions()):
            if tr[0] in states_output or (not state_info and not reward_info):
                output.write(f"\t{tr[0]} -- {tr[1]} --> {tr[2]}\n")
            else:
                states_output.add(tr[0])
                r = self.get_reward(tr[0])
                i = self._initial_probabilities[tr[0]]
                if state_info and reward_info and r!=0:
                    output.write(f"\t{tr[0]} [p: {i}; r: {r}] -- {tr[1]} --> {tr[2]}\n")
                elif state_info and (not reward_info or r==0):
                    output.write(f"\t{tr[0]} [p: {i}] -- {tr[1]} --> {tr[2]}\n")
                elif not state_info and (reward_info or r!=0):
                    output.write(f"\t{tr[0]} [r: {r}] -- {tr[1]} --> {tr[2]}\n")

        output.write("}\n")

        result = output.getvalue()
        output.close()
        return result

    def add_state(self, s: str):
        '''Add a state named s to the MC'''
        # check if it already exists or not
        if not s in self._states:
            self._states.append(s)

    def number_of_states(self)->int:
        '''Return the number of states.'''
        return len(self._states)

    def states(self)->List[str]:
        '''Return the list of states.'''
        return self._states

    def sort_state_names(self):
        '''Sort the list of states.'''
        self._states = sort_names(self._states)
        self._initial_probability_vector = None
        self._transition_matrix = None

    def set_initial_probability(self, s: str, p: Fraction):
        '''Set initial probability of state s to p.'''
        if not s in self._states:
            raise DTMCException('Unknown state')
        self._initial_probabilities[s] = p

    def set_reward(self, s: str, r: Fraction):
        '''Set reward of state s to r.'''
        if not s in self._states:
            raise DTMCException('Unknown state')
        self._rewards[s] = r

    def get_reward(self, s: str)->Fraction:
        '''Get reward of state s. Defaults to 0 if undefined.'''
        if not s in self._rewards:
            return Fraction(0)
        return self._rewards[s]

    def set_edge_probability(self, s: str, d: str, p: Fraction):
        '''Set the probability of the transition from s to d to p.'''
        if not s in self._states or not d in self._states:
            raise DTMCException('Unknown state')
        if s not in self._transitions:
            self._transitions[s] = {}
        self._transitions[s][d] = p

    def transitions(self)->Set[Tuple[str,Fraction,str]]:
        '''Get the transitions of the dtmc as tuples (s, p, d). With source
        state s, destination state d and probability p.'''
        result = set()
        for src_state, trans in self._transitions.items():
            for (dst_state, p) in trans.items():
                result.add((src_state, p, dst_state))
        return result

    def add_implicit_transitions(self):
        '''Add the implicit transitions when the outgoing probabilities do not add up to one.'''
        # compute the transition matrix, which will have the implicit transition probabilities
        matrix: linalg.TMatrix = self.transition_matrix()
        # set all transitions according to the non-zero elements in the matrix
        n_states = len(self._states)
        for i in range(n_states):
            si = self._states[i]
            for j in range(n_states):
                sj = self._states[j]
                if not matrix[j][i]==Fraction(0):
                    # add the transition if it does not yet exist
                    if not si in self._transitions:
                        self.set_edge_probability(si, sj, matrix[j][i])
                    else:
                        if not sj in self._transitions[si]:
                            self.set_edge_probability(si, sj, matrix[j][i])

    def _complete_transition_matrix(self):
        '''Complete the transition matrix with missing/implicit transitions.'''
        if self._transition_matrix is None:
            raise DTMCException("Transition matrix is not yet initialized.")
        # ensure that all rows add up to 1.
        # compute the row sums
        sums = linalg.row_sum(self._transition_matrix)
        nr_states = len(self._states)
        for n in range(nr_states):
            # if the row n sum is significantly smaller than 1
            if sums[n] < Fraction(999, 1000):
                # try to add the missing probability mass on a self-loop on n,
                # if it is not specified (is zero)
                if self._transition_matrix[n][n] == Fraction(0):
                    self._transition_matrix[n][n] = Fraction(1) - sums[n]
                else:
                    # cannot interpret it as an implicit transition
                    raise DTMCException("probabilities do not add to one")
            else:
                if sums[n] < Fraction(1001, 1000):
                    # It is almost one, fix the largest probability
                    k_max = 0
                    max_val = Fraction(0)
                    for k in range(nr_states):
                        if self._transition_matrix[k][n] > max_val:
                            max_val = self._transition_matrix[k][n]
                            k_max = k
                    # determine the correct value for self._transitionMatrix[kMax][n]
                    mass = Fraction(1)
                    for k in range(nr_states):
                        if k != k_max:
                            mass = mass - self._transition_matrix[k][n]

                    self._transition_matrix[k_max][n] = mass

                else:
                    # it is significantly larger than 1
                    raise DTMCException("probability mass is larger than one")

    def transition_matrix(self)->linalg.TMatrix:
        '''Computes and returns the transition matrix of the MC.'''
        n_states = len(self._states)
        self._transition_matrix = linalg.zero_matrix(n_states, n_states)
        row = 0
        for ss in self._states:
            if ss in self._transitions:
                col = 0
                for sd in self._states:
                    if sd in self._transitions[ss]:
                        self._transition_matrix[col][row] = self._transitions[ss][sd]
                    col += 1
            row += 1
        self._complete_transition_matrix()
        return self._transition_matrix

    def initial_probability_specified(self, s: str)->bool:
        '''Return if state s has a specified initial probability.'''
        return s in self._initial_probabilities

    def reward_specified(self, s: str)->bool:
        '''Return if state s has a specified reward.'''
        return s in self._rewards

    def complete_initial_probability_vector(self):
        '''Complete the vector of initial probabilities with implicit probabilities.'''
        if self._initial_probability_vector is None:
            raise DTMCException("Initial probability vector is not yet initialized.")
        # ensure that the initial probabilities add up to 1.
        v_sum = linalg.vector_sum(self._initial_probability_vector)
        if v_sum > Fraction(1):
            raise DTMCException("probability is larger than one")
        if v_sum < Fraction(1):
            k_ns = [self.initial_probability_specified(s) for s in self._states].count(False)
            if k_ns == 0:
                raise DTMCException("probability mass is smaller than one")
            remainder: Fraction = (Fraction(1) - v_sum) / Fraction(k_ns)
            k = 0
            for s in self._states:
                if not self.initial_probability_specified(s):
                    self._initial_probability_vector[k] = remainder
                k += 1

    def complete_initial_probabilities(self):
        '''Complete the initial probabilities.'''
        # ensure that the initial probabilities add up to 1.
        sum_value = functools.reduce(lambda a,b : a+b, \
                        self._initial_probabilities.values(), Fraction(0))
        if sum_value > Fraction(1001,1000):
            # probability is significantly larger than 1
            raise DTMCException("initial probability mass is larger than one")
        if sum_value < Fraction(999, 1000):
            k_ns = [self.initial_probability_specified(s) for s in self._states].count(False)
            if k_ns == 0:
                raise DTMCException("initial probability mass is smaller than one")
            remainder: Fraction = (Fraction(1) - sum_value) / Fraction(k_ns)
            k = 0
            for s in self._states:
                if not self.initial_probability_specified(s):
                    self.set_initial_probability(s, remainder)
                k += 1
        else:
            # probability is close to one, but possibly not exactly equal
            # fix the largest probability

            s_max = ""
            max_val = Fraction(0)
            for s in self._states:
                if self.initial_probability_specified(s):
                    if self._initial_probabilities[s] > max_val:
                        max_val = self._initial_probabilities[s]
                        s_max = s
            # determine the correct value for self._initialProbabilities[s]
            mass = Fraction(1)
            for s in self._states:
                if s != s_max:
                    if self.initial_probability_specified(s):
                        mass = mass - self._initial_probabilities[s]

            self.set_initial_probability(s_max, mass)



    def complete_rewards(self):
        '''Complete the implicit rewards to zero.'''
        # Initialize reward zero if not defined in dtmc model
        for s in self._states:
            if not self.reward_specified(s):
                self.set_reward(s, Fraction(0))

    def initial_probability_vector(self)->linalg.TVector:
        '''Determine and return the initial probability vector.'''
        n_states = len(self._states)
        self._initial_probability_vector = linalg.zero_vector(n_states)
        k = 0
        for s in self._states:
            if s in self._initial_probabilities:
                self._initial_probability_vector[k] = self._initial_probabilities[s]
            k += 1
        self.complete_initial_probability_vector()
        return self._initial_probability_vector

    def reward_vector(self)->linalg.TVector:
        '''Return reward vector.'''
        return [self.get_reward(s) for s in self._states]

    def reward_for_distribution(self, d: linalg.TVector)->Fraction:
        '''Return expected reward for a given distribution.'''
        result = Fraction(0)
        for k in range(self.number_of_states()):
            result += d[k] * self.get_reward(self._states[k])
        return result

    def execute_steps(self, n_steps: int)->List[linalg.TVector]:
        '''Perform n_steps steps of the chain, return array of N+1
        distributions, starting with the initial distribution and
        distributions after N steps.'''
        if n_steps<0:
            raise DTMCException("Number of steps must be non-negative.")
        p_matrix = self.transition_matrix()
        pi = self.initial_probability_vector()
        result = []
        for _ in range(n_steps+1):
            result.append(pi)
            pi = linalg.vector_matrix_product(pi, p_matrix)
        return result


    def _compute_communication_graph(self)->pyg.digraph:
        '''Return the communication graph corresponding to the Markov Chain.'''
        gr = pyg.digraph()
        gr.add_nodes(self._states)
        for s, trans in self._transitions.items():
            for t, p in trans.items():
                if not p == Fraction(0):
                    gr.add_edge((s, t))
        return gr

    def _compute_reverse_communication_graph(self)->pyg.digraph:
        '''Return the communication graph corresponding to the reversed
        transitions of the Markov Chain.'''
        gr = pyg.digraph()
        gr.add_nodes(self._states)
        for s, trans in self._transitions.items():
            for t, p in trans.items():
                if not p==Fraction(0):
                    gr.add_edge((t, s))
        return gr

    def communicating_classes(self)->Set[AbstractSet[str]]:
        '''Determine the communicating classes of the dtmc. Returns Set of sets
        of states of the dtmc.'''
        gr = self._compute_communication_graph()
        return {frozenset(s) for s in pyga.mutual_accessibility(gr).values()}

    def classify_transient_recurrent_classes(self)->\
        Tuple[Set[AbstractSet[str]],Set[AbstractSet[str]]]:
        '''Classify the states into transient and recurrent. Return a pair with
        transient classes and recurrent classes.'''
        c_classes = self.communicating_classes()
        r_classes = c_classes.copy()
        state_map = {}
        for c in c_classes:
            for s in c:
                state_map[s] = c

        # remove all classes with outgoing transitions
        for s, trans in self._transitions.items():
            if state_map[s] in r_classes:
                for t in trans:
                    if state_map[t] != state_map[s]:
                        if state_map[s] in r_classes:
                            r_classes.remove(state_map[s])

        return c_classes, r_classes

    def classify_transient_recurrent(self)->Tuple[Set[str],Set[str]]:
        '''Classify states into transient and recurrent.'''
        _, r_classes = self.classify_transient_recurrent_classes()

        # collect all recurrent states
        r_states = set()
        for c in r_classes:
            r_states.update(c)

        # remaining states are transient
        t_states = set(self._states).difference(r_states)

        return t_states, r_states

    def classify_periodicity(self)->Dict[str,int]:
        '''Determine periodicity of states. Returns a dictionary mapping state to periodicity.'''

        def _cycle_found(k:str):
            nonlocal node_stack, cycles_found
            i = node_stack.index(k)
            cycles_found.add(frozenset(node_stack[i:len(node_stack)]))

        def _explore_cycles(m: str):
            explored_nodes.add(m)
            node_stack.append(m)
            for k in gr.neighbors(m):
                if k in node_stack:
                    _cycle_found(k)
                else:
                    _explore_cycles(k)
            node_stack.pop(len(node_stack)-1)

        self.add_implicit_transitions()

        gr = self._compute_communication_graph()
        # perform exploration for all states
        cycles_found:Set[AbstractSet[str]] = set()
        explored_nodes: Set[str] =set()
        nodes_to_explore: List[str] = list(gr.nodes())
        node_stack: List[str] = []
        while len(nodes_to_explore) > 0:
            n: str = nodes_to_explore.pop(0)
            if not n in explored_nodes:
                _explore_cycles(n)

        # compute periodicities of the recurrent states
        # periodicity is gcd of the length of all cycles reachable from the state
        _, r_states = self.classify_transient_recurrent()

        per: Dict[str,int] = {}
        for c in cycles_found:
            cl = len(c)
            for s in c:
                if s in r_states:
                    if not s in per:
                        per[s] = cl
                    else:
                        per[s] = math.gcd(cl, per[s])

        comm_classes = self.communicating_classes()
        for cl in comm_classes:
            s = next(iter(cl))
            if s in r_states:
                p = per[s]
                for s in cl:
                    p = math.gcd(p, per[s])
                for s in cl:
                    per[s] = p

        return per

    def determine_mc_type(self)->Literal['ergodic unichain', \
                'non-ergodic unichain','ergodic non-unichain','non-ergodic non-unichain']:
        '''Return the type of the MC.
        A class that is both recurrent and aperiodic is called an ergodic class.
        A Markov chain having a single class of communicating states is called
        an irreducible Markov chain. Notice that this class of states is
        necessarily recurrent. In case this class is also aperiodic, i.e. if
        the class is ergodic, the chain is called an ergodic Markov chain.
        #A Markov chain that contains a single recurrent class in addition to
        zero or more transient classes, is called a unichain. In case the
        recurrent class is ergodic, we speak about an ergodic unichain. A
        unichain visits its transients states a finite number of times, after
        which the chain enters the unique class of recurrent states in which it
        remains for ever.
        '''

        _, r_classes =  self.classify_transient_recurrent_classes()
        per = self.classify_periodicity()

        is_unichain = len(r_classes) == 1
        e_classes = [c for c in r_classes if per[next(iter(c))] == 1]

        if is_unichain:
            if len(e_classes) > 0:
                return 'ergodic unichain'
            return 'non-ergodic unichain'
        if len(r_classes) == len(e_classes):
            return 'ergodic non-unichain'

        return 'non-ergodic non-unichain'

    def _hitting_probabilities(self, target_state: str)->Tuple[List[str], \
                    Optional[linalg.TMatrix],Dict[str,int],Dict[str,int],Dict[str,Fraction]]:
        '''Determine the hitting probabilities to hit targetState. Returns a tuple with:
        - rs: the list of states from which the target state is reachable
        - ImEQ: the matrix of the matrix equation
        - rIndex: index numbers of the states from rs in the equation
        - pIndex, index of all states
        - res: the hitting probabilities
        '''

        def _states_reachable_from(s: str)->List[str]:
            _, pre, _ = pygs.depth_first_search(gr, root=s)
            return pre

        # determine the set of states from which targetState is reachable
        gr = self._compute_reverse_communication_graph()
        rs = _states_reachable_from(target_state)
        rs = list(rs)
        # give each of the states an index in map rIndex
        r_index:Dict[str,int] = {}
        for k, s in enumerate(rs):
            r_index[s] = k

        # determine transition matrix and state indices
        p_matrix = self.transition_matrix()
        p_index:Dict[str,int] = {}
        for k, s in enumerate(self._states):
            p_index[s] = k
        # jp is index of the target state
        jp = p_index[target_state]

        i_minus_eq = None
        sol_x: Optional[linalg.TVector] = None

        if  len(rs) > 0:

            # determine the hitting prob equations
            # Let us fix a state j ∈ S and define for each state i in rs a
            # corresponding variable x_i (representing the hit probability
            # f_ij ). Consider the system of linear equations defined by
            # x_i = Pij + Sum _ k∈S\{j} P_ik x_k
            # solve the equation: x = EQ x + Pj
            # solve the equation: (I-EQ) x = Pj

            n_rs = len(rs)
            # initialize matrix I-EQ from the equation, and vector Pj
            i_minus_eq = linalg.identity_matrix(n_rs)
            pj = linalg.zero_vector(n_rs)
            # for all equations (rows of the matrix)
            for i in range(n_rs):
                ip = p_index[rs[i]]
                pj[i] = p_matrix[jp][ip]
                # for all variables in the summand
                for k in range(n_rs):
                    # note that the sum excludes the target state!
                    if rs[k] != target_state:
                        kp = p_index[rs[k]]
                        # determine row i, column k
                        i_minus_eq[k][i] -= p_matrix[kp][ip]

            # solve the equation x = inv(I-EQ)*Pj
            sol_x = linalg.solve(i_minus_eq, pj)

        # set all hitting probabilities to zero
        res: Dict[str,Fraction] = {}
        for s in self._states:
            res[s] = Fraction(0)

        # fill the solutions from the equation
        for s in rs:
            sol_xv: linalg.TVector = sol_x  # type: ignore we know sol_x is a vector
            res[s] = sol_xv[r_index[s]]

        return rs, i_minus_eq, r_index, p_index, res


    def hitting_probabilities(self, target_state: str)->Dict[str,Fraction]:
        '''Determine the hitting probabilities to hit targetState.'''

        _, _, _, _, res = self._hitting_probabilities(target_state)
        return res

    def reward_till_hit(self, target_state: str)->Dict[str,Fraction]:
        '''Determine the expected reward until hitting targetState'''

        rs, i_minus_eq, r_index, _, f = self._hitting_probabilities(target_state)
        sol_x = None
        if  len(rs) > 0:

            # x_i = r(i) · fij + Sum k∈S\{j} P_ik x_k
            # solve the equation: x = EQ x + Fj
            # solve the equation: (I-EQ) x = Fj

            n_states = len(rs)
            fj = linalg.zero_vector(n_states)
            for i in range(n_states):
                si = rs[i]
                fj[i] = self.get_reward(si) * f[si]

            # solve the equation x = inv(I-EQ)*Fj
            i_minus_eq_m: linalg.TMatrix = i_minus_eq  # type: ignore we know ImEQ is a matrix
            sol_x = linalg.solve(i_minus_eq_m, fj)

        # set fr to zero
        fr: Dict[str,Fraction] = {}
        for s in self._states:
            fr[s] = Fraction(0)

        # fill the solutions from the equation
        for s in rs:
            sol_xv: linalg.TVector = sol_x  # type: ignore we know solX is a vector
            fr[s] = sol_xv[r_index[s]]

        res = {}
        for s in rs:
            res[s] = fr[s] / f[s]

        return res


    def _hitting_probabilities_set(self, target_states: List[str])-> \
        Tuple[linalg.TMatrix,List[str],Optional[linalg.TMatrix],Dict[str,int], \
        Dict[str,int],Dict[str,Fraction]]:
        '''Determine the hitting probabilities to hit a set targetStates. Returns a tuple with:

        - P: the transition matrix
        - rs: the list of states from which the target state is reachable
        - ImEQ: the matrix of the matrix equation
        - rIndex: index numbers of the states from rs in the equation
        - pIndex, index of all states
        - res: the hitting probabilities
        '''

        def _states_reachable_from(s: str)->List[str]:
            _, pre, _ = pygs.depth_first_search(gr, root=s)
            return pre

        # determine the set of states from which the set targetStates are reachable
        # not very efficient...
        gr = self._compute_reverse_communication_graph()
        rs = set()
        for s in target_states:
            rs.update(_states_reachable_from(s))

        # exclude target states
        rs = rs.difference(target_states)

        # fix an arbitrary order
        rs = list(rs)

        # make an index on rs
        r_index = {}
        for k, s in enumerate(rs):
            r_index[s] = k

        # get transition matrix
        p_matrix = self.transition_matrix()
        # make index on P matrix
        p_index = {}
        for k, s in enumerate(self._states):
            p_index[s] = k

        # determine the hitting prob equations
        # x_i = 0                       if i in S \ (rs U targetStates)
        # x_i = 1                       if i in targetStates
        # x_i = sum _ k in rs P_ik x_k + sum _ k in targetStates P_ik       if i in rs
        # equation for the third case, take first two cases as constants:
        # x_i = sum _ k in rs P_ik x_k + SP(i) = sum _ k in targetStates P_ik       for i in rs

        # solve the equation: x = EQ x + Pj
        # solve the equation: (I-EQ) x = Pj

        n_states = len(rs)
        i_minus_eq = linalg.identity_matrix(n_states)
        sp = linalg.zero_vector(n_states)
        for i in range(n_states):
            ip = p_index[rs[i]]
            # compute the i-th element in vector SP
            for s in target_states:
                sp[i] += p_matrix[p_index[s]][ip]

            # compute the i-th row in matrix ImEQ
            for k in range(n_states):
                kp = p_index[rs[k]]
                # determine row i, column k
                i_minus_eq[k][i] -= p_matrix[kp][ip]

        # solve the equation x = inv(I-EQ)*SP
        sol_x = linalg.solve(i_minus_eq, sp)

        # set all hitting probabilities to zero
        res: Dict[str,Fraction] = {}
        for s in self._states:
            res[s] = Fraction(0)

        # set all hitting probabilities in the target set to 1
        for s in target_states:
            res[s] = Fraction(1)

        # fill the solutions from the equation
        for s in rs:
            res[s] = sol_x[r_index[s]]

        return p_matrix, rs, i_minus_eq, r_index, p_index, res

    def hitting_probabilities_set(self, target_states: List[str])->Dict[str,Fraction]:
        '''Determine the hitting probabilities to hit a set targetStates.'''
        _, _, _, _, _, res = self._hitting_probabilities_set(target_states)
        return res

    def reward_till_hit_set(self, target_states: List[str]):
        '''Determine the expected reward until hitting set targetStates.'''

        _, rs, i_minus_eq, r_index, _, h = self._hitting_probabilities_set(target_states)

        sol_x = None

        if  len(rs) > 0:

            # x_i = sum _ k in rs P_ik x_k  +  hh(i) = r(i) · h_i    for all i in rs

            # solve the equation: x = EQ x + H
            # solve the equation: (I-EQ) x = H

            n_states = len(rs)
            hh = linalg.zero_vector(n_states)
            for i in range(n_states):
                # compute the i-th element in vector H
                si = rs[i]
                hh[i] = self.get_reward(si) * h[si]

            # solve the equation x = inv(I-EQ)*H
            i_minus_eq_m: linalg.TMatrix = i_minus_eq  # type: ignore we know that ImEQ is a matrix
            sol_x = linalg.solve(i_minus_eq_m, hh)

        # set hr to zero
        hr: Dict[str,Fraction] = {}
        for s in self._states:
            hr[s] = Fraction(0)

        # fill the solutions from the equation
        for s in rs:
            sol_xv: linalg.TVector = sol_x  # type: ignore we know that solX is a vector
            hr[s] = sol_xv[r_index[s]]

        res: Dict[str,Fraction] = {}
        for s in target_states:
            res[s] = Fraction(0)
        for s in rs:
            res[s] = hr[s] / h[s]

        return res

    def _get_sub_transition_matrix_indices(self, indices: List[int])->linalg.TMatrix:
        '''Return sub transition matrix consisting of the given list of indices.'''
        if self._transition_matrix is None:
            raise DTMCException("Transition matrix has not been determined.")
        n_indices = len(indices)
        res = linalg.zero_matrix(n_indices, n_indices)
        for k in range(n_indices):
            for m in range(n_indices):
                res[k][m] = self._transition_matrix[indices[k]][indices[m]]
        return res

    def _get_sub_transition_matrix_class(self, cl: AbstractSet[str])-> \
        Tuple[Dict[str,int],linalg.TMatrix]:
        '''Return an index for the states in C and a sub transition matrix for the class C.'''
        # get sub-matrix for a class C of states
        indices = sorted([self._states.index(s) for s in cl])
        index = {c:indices.index(self._states.index(c)) for c in cl}
        return index, self._get_sub_transition_matrix_indices(indices)

    def limiting_matrix(self)->linalg.TMatrix:
        '''Determine the limiting matrix of the dtmc.'''
        # formulate and solve balance equations for each of the  recurrent classes
        # determine the recurrent classes
        self.transition_matrix()

        n_states = len(self._states)
        res = linalg.zero_matrix(n_states, n_states)

        _, r_classes =  self.classify_transient_recurrent_classes()

        # for each recurrent class:
        for c in r_classes:
            index, p_matrix_cl = self._get_sub_transition_matrix_class(c)
            # a) solve the balance equations, pi P = pi I , pi.1 = 1
            #       pi (P-I); 1 = [0 1],
            m = len(c)
            p_minus_i = linalg.subtract_matrix(p_matrix_cl, linalg.identity_matrix(m))
            q_matrix = linalg.add_matrix(linalg.matrix_matrix_product(p_minus_i, \
                            linalg.transpose(p_minus_i)), linalg.one_matrix(m,m))

            q_inverse = linalg.invert_matrix(q_matrix)
            pi = linalg.column_sum(q_inverse)
            h = self.hitting_probabilities_set(list(c))
            # P(i,j) = h_i * pi j
            for sj in c:
                j = self._states.index(sj)
                for i in range(n_states):
                    if self._states[i] in c:
                        res[j][i] = pi[index[sj]]
                    else:
                        res[j][i] = h[self._states[i]] * pi[index[sj]]
        return res

    def limiting_distribution(self)->linalg.TVector:
        '''Determine the limiting distribution.'''
        p_matrix = self.limiting_matrix()
        pi0 = self.initial_probability_vector()
        return linalg.vector_matrix_product(pi0, p_matrix)

    def long_run_reward(self)-> Fraction:
        '''Determine the long-run expected reward.'''
        pi = self.limiting_distribution()
        r = self.reward_vector()
        return linalg.inner_product(pi, r)

    @staticmethod
    def from_dsl(dsl_string: str)->Tuple[Optional[str],Optional['MarkovChain']]:
        """Construct from a DSL specification."""

        factory = {}
        factory['Init'] = MarkovChain
        factory['AddState'] = lambda dtmc, s: (dtmc.add_state(s), s)[1]
        factory['SetInitialProbability'] = lambda dtmc, s, p: dtmc.set_initial_probability(s, p)
        factory['SetReward'] = lambda dtmc, s, r: dtmc.set_reward(s, r)
        factory['SetEdgeProbability'] = lambda dtmc, s, d, p: dtmc.set_edge_probability(s, d, p)
        factory['SortNames'] = lambda dtmc: dtmc.sort_state_names()
        return parse_dtmc_dsl(dsl_string, factory)

    def __str__(self)->str:
        return str(self._states)


    # ---------------------------------------------------------------------------- #
    # - Markov Chain simulation                                                  - #
    # ---------------------------------------------------------------------------- #

    _stopDescriptions = [STR_ABS_ERROR, STR_REL_ERROR, STR_NR_OF_PATHS, STR_TIMEOUT]

    def set_seed(self, seed: int):
        ''' Set random generator seed'''
        random.seed(seed)

    def random_initial_state(self)->str:
        '''Return random initial state according to initial state distribution'''
        r = random.random()
        p: float = 0.0

        for s, pr in self._initial_probabilities.items():
            p = p + pr
            if r < p:
                return s
        # probability 0 of falling through to this point
        return self._states[0]

    def random_transition(self, s: str)->str:
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


    def set_recurrent_state(self, state:Optional[str]):
        '''
        Set the recurrent state for simulation. If the given state is not a recurrent state,
        an exception is raised. If state is None, recurrent state is cleared.
        '''
        if state is None:
            self._recurrent_state = None
        else:
            if self._is_recurrent_state(state):
                self._recurrent_state = state
            else:
                raise DTMCException(f"{state} is not a recurrent state.")

    def _markov_simulation(self, actions: List[TSimulationActionAndDescription], \
                           initial_state: Optional[str] = None)->Tuple[int,Optional[str]]:
        '''
        Simulate Markov Chain.
        actions is a list of pairs consisting of a callable that is called upon every step of the
        simulation and an optional string that describes the reason why the simulation ends.
        The callable should take two arguments: n: int, the number of performed simulation steps
        before this one, and state: str, the current state of the Markov Chain in the simulation.
        It should return a Boolean value indicating if the simulation should be ended.
        An optional forced initial state can be provided. If no initial state is provided, it is
        selected randomly according to the initial state distribution.
        Returns a pair n, stop, consisting of the total number of steps simulated and the optional
        string describing the reason for stopping.
        '''
        # list for stop condition status
        stop_conditions:List[bool] = [False] * len(actions)

        # Step counter
        n: int = 0

        # Determine current state as random initial state
        if initial_state is None:
            current_state: Optional[str] = self.random_initial_state()
        else:
            current_state = initial_state

        while not any(stop_conditions):
            # perform simulation actions
            for i, (action,_) in enumerate(actions):
                stop_conditions[i] = action(n, current_state)

            # next random step in Markov Chain simulation
            current_state = self.random_transition(current_state)
            n += 1

        # Determine stop condition
        stop = None
        for i, (_,st) in enumerate(actions):
            if stop_conditions[i]:
                stop = st

        return n, stop


    def markov_simulation_recurrent_cycles(self, perm_actions: List \
        [TSimulationActionAndDescription], cycle_actions: List[TSimulationActionAndDescription], \
        complete_cycle: Callable[[],bool], initial_state: Optional[str] = None):
        '''
        simulate a uni-chain MC from recurrent, provide callback on every state and on every visit
        of the recurrent state.
        If _recurrentState is defined, it will be use as the recurrent state; otherwise the first
        visited recurrent state is used.
        '''

        recurrent_state_visits: int = 0
        recurrent_states: Set[str] = (self.classify_transient_recurrent())[1]

        def _action_cycle_update(_:int, state:str)->bool:
            nonlocal recurrent_state_visits, recurrent_states
            if self._recurrent_state is None:
                # we have no designated recurrent state yet
                if state in recurrent_states:
                    # if the current state is a recurrent state, use it
                    self._recurrent_state = state
                    recurrent_state_visits = 1
                return False
            # we have a designated recurrent state
            if state != self._recurrent_state:
                # the current state is not it
                return False
            # we visit the recurrent state
            if recurrent_state_visits > 0:
                # we have visited it before
                recurrent_state_visits += 1
                return complete_cycle()
            # mark it as visited and carry on
            recurrent_state_visits = 1
            return False

        def _in_recurrent_cycle_do(a: Callable[[int,str],bool], n, state)->bool:
            if recurrent_state_visits > 0:
                return a(n, state)
            return False

        # make a combined list of permanent actions, in-cycle actions and cycle update action
        _sim_actions: List[TSimulationActionAndDescription] = perm_actions
        # note the python magic 'a=a' in the lambda function to make sure a gets bound to
        # its current value
        _sim_actions.extend([(lambda n, state, a=a: _in_recurrent_cycle_do(a[0], n, state), \
                              a[1]) for a in cycle_actions])
        _sim_actions.append((_action_cycle_update, "Number of cycles"))

        return self._markov_simulation(_sim_actions, initial_state)


    def _is_recurrent_state(self, state: str)->bool:
        '''Check if state is a recurrent state. Note that this method may be time consuming as
        an analysis is performed every time it is called.'''
        # recurrent state thats encountered in the random state_sequence dictionary
        _, r_classes = self.classify_transient_recurrent()
        return state in r_classes

    def markov_trace(self, rounds: int)->List[str]:
        '''Simulate Markov Chain for the given number of rounds (steps).
        Returns a list of length rounds+1 of the states visited in the simulation. '''
        # Declare empty list for states
        state_list: List[str] = []

        def _collect_list(_: int, state: str)->bool:
            state_list.append(state)
            return False

        self._markov_simulation([
            (lambda n, _: n >= rounds, None), # Exit when n is number of rounds
            (_collect_list, None) # add state to list
        ])

        return state_list

    def generic_long_run_sample_estimation_from_recurrent_cycles(self, sc:StopConditions, \
            action_update: Callable[[Statistics,int,str],bool])->Tuple[Statistics, Optional[str]]:
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
        # Global variables used during simulation
        # statistics = Statistics(sc.confidence, sc.minimumNumberOfSamples)
        statistics = Statistics(sc.confidence)

        def _action_abs_err(_n:int, _state:str)->bool:
            c = statistics.ab_error()
            if c is None:
                return False
            return sc.max_ab_error > 0 and c <= sc.max_ab_error

        def _action_rel_err(_n:int, _state:str)->bool:
            c = statistics.re_error()
            if c is None:
                return False
            return sc.max_re_error > 0 and c <= sc.max_re_error

        def _cycle_update()->bool:
            statistics.complete_cycle()
            return 0 <= sc.nr_of_cycles <= statistics.cycle_count()

        def _action_update(n: int, state: str)->bool:
            return action_update(statistics, n, state)

        t = TimeoutTimer(sc.seconds_timeout)
        _, stop = self.markov_simulation_recurrent_cycles(
            [
                # Run until max path length has been reached
                (lambda n, _: 0 <= sc.max_path_length <= n, STR_MAX_PATH_LENGTH),
                (t.sim_action(), STR_TIMEOUT), # Exit on time out
            ], [
                (_action_update, None),
                (_action_abs_err, STR_ABS_ERROR), # check absolute error
                (_action_rel_err, STR_REL_ERROR) # check relative error
            ],
            _cycle_update)
        return statistics, stop

    def generic_long_run_distribution_estimation_from_recurrent_cycles(self, sc:StopConditions, \
            action_update: Callable[[DistributionStatistics,int,str],bool])-> \
                Tuple[DistributionStatistics, Optional[str]]:
        '''generic long-run distribution estimation framework by simulation using the provided
        stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        Returns a tuple: each of the results can be None if they could not be determined.
        - DistributionStatistics with:
            - confidence interval
            - estimate of the absolute error
            - estimate of the relative error
            - point estimate of the mean
            - point estimate of the variance
        - the stop criterion applied as a string
        '''

        distribution_statistics = DistributionStatistics(self.number_of_states(), sc.confidence)

        def _complete_cycle()->bool:
            distribution_statistics.complete_cycle()
            return 0 <= sc.nr_of_cycles <= distribution_statistics.cycle_count()

        def _action_visit_state(n:int, state:str)->bool:
            return action_update(distribution_statistics, n, state)

        def _action_ab_error(_n: int, _state:str)->bool:
            c = distribution_statistics.ab_error()
            if c is None:
                return False
            if any(v is None for v in c):
                return False
            vc : List[float] = c  # type: ignore
            return sc.max_ab_error > 0 and max(vc) <= sc.max_ab_error

        def _action_re_error(_n: int, _state:str)->bool:
            c = distribution_statistics.re_error()
            if c is None:
                return False
            if any(v is None for v in c):
                return False
            vc : List[float] = c  # type: ignore
            return sc.max_re_error > 0 and max(vc) <= sc.max_re_error

        t = TimeoutTimer(sc.seconds_timeout)
        _, stop = self.markov_simulation_recurrent_cycles(
            [
                # Run until max path length has been reached
                (lambda n, state: 0 <= sc.max_path_length <= n, STR_MAX_PATH_LENGTH),
                (t.sim_action(), STR_TIMEOUT), # Exit on time
            ], [
                (_action_visit_state, ""),
                (_action_ab_error, STR_ABS_ERROR), # Calculate smallest absolute error
                (_action_re_error, STR_REL_ERROR) # Calculate smallest relative error
            ],
            _complete_cycle)

        return distribution_statistics, stop

    def long_run_expected_average_reward(self, sc:StopConditions)->Tuple[Statistics, Optional[str]]:
        '''Estimate the long run expected average reward by simulation using the provided
        stop_conditions.
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
        def _action_add_sample(statistics: Statistics, _n:int, state:str)->bool:
            statistics.add_sample(float(self._rewards[state]))
            return False

        return self.generic_long_run_sample_estimation_from_recurrent_cycles(sc, _action_add_sample)

    def cezaro_limit_distribution(self, sc:StopConditions)-> Tuple[Optional \
                    [DistributionStatistics], Optional[str]]:
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

        def action_visit_state(s: DistributionStatistics, _:int, state:str)->bool:
            s.visit_state(self._states.index(state))
            return False

        return self.generic_long_run_distribution_estimation_from_recurrent_cycles(sc, \
                                                                action_visit_state)


    def estimation_generic_transient_sample(self, sc:StopConditions, get_sample: Callable[[str], \
                    float], nr_of_steps)->Tuple[Statistics, Optional[str]]:
        '''
        Estimate the transient sample after nr_of_steps by simulation using the provided
        stop_conditions.
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

        statistics = Statistics(sc.confidence, sc.minimum_number_of_samples)

        def _action_last_state_reward(n: int, state: str)->bool:
            if n == nr_of_steps:
                statistics.add_sample(get_sample(state))
                statistics.complete_cycle()
            return False

        sim_stop_conditions: List[bool] = [False] * 4

        t = TimeoutTimer(sc.seconds_timeout)
        while not any(sim_stop_conditions):
            self._markov_simulation([
                (_action_last_state_reward, None),
                (lambda n, _: 0 <= nr_of_steps <= n, None), # Exit when n is number of rounds
                (t.sim_action(), STR_TIMEOUT), # Exit on time
            ])

            # Check stop conditions
            sim_stop_conditions[0] = statistics.ab_error_reached(sc.max_ab_error)
            sim_stop_conditions[1] = statistics.re_error_reached(sc.max_re_error)
            sim_stop_conditions[2] = 0 <= sc.nr_of_cycles <= statistics.cycle_count()
            sim_stop_conditions[3] = t.is_expired()

        # Determine stop condition
        stop = self._stopDescriptions[sim_stop_conditions.index(True)]
        return statistics, stop

    def estimation_expected_reward(self, sc:StopConditions, nr_of_steps)->Tuple[
        Statistics,
        Optional[str]]:
        '''
        Estimate the transient expected reward after nr_of_steps by simulation using the provided
        stop_conditions.
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

        return self.estimation_generic_transient_sample(sc, lambda state: float( \
            self.get_reward(state)), nr_of_steps)


    def estimation_transient_distribution(self, sc:StopConditions, nr_of_steps: int)->Tuple[
        DistributionStatistics, Optional[str]]:
        '''
        Estimate the distribution after nr_of_steps by simulation using the provided
        stop_conditions.
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

        distribution_statistics = DistributionStatistics(len(self._states), sc.confidence)

        # There are in total four applicable stop conditions for this function
        sim_stop_conditions: List[bool] = [False] * 4

        t = TimeoutTimer(sc.seconds_timeout)
        current_state: Optional[str] = None

        def _action_track_state(_n: int, state: str)-> bool:
            nonlocal current_state
            current_state = state
            return False

        while not any(sim_stop_conditions):

            self._markov_simulation([
                (_action_track_state, None),
                (lambda n, state: 0 <= nr_of_steps <= n, None), # Exit when n is number of steps
                (t.sim_action(), STR_TIMEOUT), # Exit on time out
            ])

            v_current_state: str = current_state  # type: ignore
            distribution_statistics.visit_state(self._states.index(v_current_state))
            distribution_statistics.complete_cycle()

            # Check stop conditions
            sim_stop_conditions[0] = distribution_statistics.ab_error_reached(sc.max_ab_error)
            sim_stop_conditions[1] = distribution_statistics.re_error_reached(sc.max_re_error)
            sim_stop_conditions[2] = 0 <= sc.nr_of_cycles <= distribution_statistics.cycle_count()
            sim_stop_conditions[3] = t.is_expired()

        # Determine stop condition
        stop = self._stopDescriptions[sim_stop_conditions.index(True)]

        return distribution_statistics, stop


    def estimation_hitting_state_generic(self, sc:StopConditions, analysis_states: List[str],\
            initialization: Callable, action: Callable[[int,str],bool], \
            on_hit: Callable[[Statistics],None], on_no_hit: Callable[[Statistics],None])-> \
                Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:

        '''
        Generic framework for estimating hitting probability, or reward until hit by simulation
        using the provided stop_conditions.
        stop_conditions is a five-tuple with:
        - confidence level
        - absolute error
        - relative error
        - path length
        - nr. cycles
        - timeout in seconds

        The analysis is performed for all initial states in analysisStates

        initialization is called every time the analysis starts for a new starting state

        action is called every step of the simulation and should return a boolean indicating if the
        simulation should be stopped because the target set is hit

        after the simulation is finished, onHit is called if the target is hit (the maximum number
        of steps is not completed in the simulation).
        onNoHit is called if the target was not hit after the maximum number of steps.

        Returns a tuple:
        - statistics of the estimated hitting probability
        - the stop criteria applied as strings
        '''

        statistics: Dict[str,Statistics] = {}
        for s in analysis_states:
            statistics[s] = Statistics(sc.confidence)

        # There are in total four applicable stop conditions for this function
        sim_stop_conditions = [False] * 4
        stop: Dict[str,str] = {}

        t = TimeoutTimer(sc.seconds_timeout)

        for initial_state in analysis_states:

            sim_stop_conditions = [False] * 4

            # generic initialization
            initialization()

            while not any(sim_stop_conditions):

                _, sim_result = self._markov_simulation([
                    # Exit when n is number of steps
                    (lambda n, _: 0 <= sc.max_path_length <= n, "steps"),
                    (t.sim_action(), STR_TIMEOUT), # Exit on time
                    (action, "hit") # stop when hitting state is found
                ], initial_state)

                statistics[initial_state].increment_paths()

                if sim_result==STR_TIMEOUT:
                    return None, STR_TIMEOUT

                if sim_result!="steps":
                    # hitting state was hit
                    on_hit(statistics[initial_state])
                else:
                    # specific
                    on_no_hit(statistics[initial_state])

                # Check stop conditions
                sim_stop_conditions[0] = statistics[initial_state].ab_error_reached(sc.max_ab_error)
                sim_stop_conditions[1] = statistics[initial_state].re_error_reached(sc.max_re_error)
                sim_stop_conditions[2] = 0 <= sc.nr_of_cycles <= \
                    statistics[initial_state].nr_paths()
                sim_stop_conditions[3] = t.is_expired()

            # Determine stop condition
            stop[initial_state] = self._stopDescriptions[sim_stop_conditions.index(True)]

        return statistics, stop



    def estimation_hitting_probability_state(self, sc:StopConditions, hitting_state: str, \
        analysis_states: List[str])->Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:

        '''
        Estimate the hitting probability until hitting a single state in one or more steps by
        simulation using the provided stop_conditions.
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

        def on_hit(s: Statistics):
            s.add_sample(1.0)
            s.complete_cycle()

        def on_no_hit(s: Statistics):
            s.add_sample(0.0)
            s.complete_cycle()

        return self.estimation_hitting_state_generic(sc, analysis_states, initialization, action, \
                                                     on_hit, on_no_hit)

    def estimation_reward_until_hitting_state(self, sc:StopConditions, hitting_state: str, \
        analysis_states: List[str])->Tuple[Optional[Dict[str,Statistics]],Union[str,Dict[str,str]]]:

        '''
        Estimate the cumulative reward until hitting a single state by simulation using the
        provided stop_conditions.
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

        accumulated_reward: float

        def initialization():
            nonlocal accumulated_reward
            accumulated_reward = 0.0

        def action(n: int, state: str)->bool:
            # define action to be performed during simulation
            nonlocal accumulated_reward
            if n==0:
                accumulated_reward += float(self.get_reward(state))
                # suppress initial state for hitting
                return False
            if state == hitting_state:
                return True
            # reward of hitting state is npt counted
            accumulated_reward += float(self.get_reward(state))
            return False

        def on_hit(s: Statistics):
            nonlocal accumulated_reward
            s.add_sample(accumulated_reward)
            accumulated_reward = 0.0
            s.complete_cycle()

        def on_no_hit(_: Statistics):
            pass

        return self.estimation_hitting_state_generic(sc, analysis_states, initialization, action, \
                                                     on_hit, on_no_hit)


    def estimation_hitting_probability_state_set(self, sc:StopConditions, hitting_states: List \
        [str], analysis_states: List[str])->Tuple[Optional[Dict[str,Statistics]], \
        Union[str,Dict[str,str]]]:

        '''
        Estimate the hitting probability until hitting a set of states in zero or more steps by
        simulation using the provided stop_conditions.
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
        def action(_: int, state: str)->bool:
            # note that for set of states, the initial states is not suppressed like in single
            # state hitting probability
            return state in hitting_states

        def on_hit(s: Statistics):
            s.add_sample(1.0)
            s.complete_cycle()

        def on_no_hit(s: Statistics):
            s.add_sample(0.0)
            s.complete_cycle()

        return self.estimation_hitting_state_generic(sc, analysis_states, initialization, action, \
                                                     on_hit, on_no_hit)

    def estimation_reward_until_hitting_state_set(self, sc:StopConditions, hitting_states: List \
            [str], analysis_states: List[str])->Tuple[Optional[Dict[str,Statistics]], \
            Union[str,Dict[str,str]]]:

        '''
        Estimate the cumulative reward until hitting a single state by simulation using the
        provided stop_conditions.
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

        accumulated_reward: float

        def initialization():
            nonlocal accumulated_reward
            accumulated_reward = 0.0

        # define action to be performed during simulation
        def action(_: int, state: str)->bool:
            nonlocal accumulated_reward
            if state in hitting_states:
                return True
            # reward of hitting state is not counted
            accumulated_reward += float(self.get_reward(state))
            return False

        def on_hit(s: Statistics):
            nonlocal accumulated_reward
            s.add_sample(accumulated_reward)
            s.complete_cycle()
            accumulated_reward = 0.0

        def on_no_hit(_: Statistics):
            pass

        return self.estimation_hitting_state_generic(sc, analysis_states, initialization, action, \
                                                     on_hit, on_no_hit)
