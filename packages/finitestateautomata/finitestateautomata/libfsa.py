import copy
from functools import reduce
from io import StringIO
import re
import sys
from typing import AbstractSet, Callable, Dict, Iterable, List, Optional,Set, Tuple
from finitestateautomata.libfsagrammar import parse_fsa_dsl


class FSAException(Exception):
    ''' Exceptions related to FSA '''
    pass

class Automaton:

    """
    Represents variants of Finite State Automata.
    """

    _epsilonSymbol:str = '#'

    # set of states
    _states: Set[str]
    # map from states to map from symbol to set of next states
    _transitions: Dict[str,Dict[str,Set[str]]]
    # map from states to set of epsilon next states
    _epsilon_transitions: Dict[str,Set[str]]
    # set of initial states
    _initial_states: Set[str]
    # set of final states
    _final_states: Set[str]
    # map from a name to a set of states forming an acceptance set
    # note the the set of final states acts as one of the generalized acceptance sets
    _generalized_acceptance_sets: Dict[str,Set[str]]

    def __init__(self):
        self._states = set()
        self._transitions = {}
        self._epsilon_transitions = {}
        self._initial_states = set()
        self._final_states = set()
        self._generalized_acceptance_sets = {}

    def add_state(self, s: str):
        '''Add a state.'''
        self._states.add(s)

    def add_states(self, set_of_states: Set[str]):
        '''Add a set of states.'''
        self._states.update(set_of_states)

    def states(self)->AbstractSet[str]:
        '''Returns the (non-modifiable) set of states.'''
        return frozenset(self._states)

    def add_transition(self, src_state: str, symbol: str, dst_state: str):
        '''Add a transition from srcState to dstState, labelled with symbol.'''
        self.add_state(src_state)
        self.add_state(dst_state)
        # update the transitions
        if not src_state in self._transitions:
            self._transitions[src_state] = {}
        if not symbol in self._transitions[src_state].keys():
            self._transitions[src_state][symbol] = set()
        self._transitions[src_state][symbol].add(dst_state)

    def has_proper_transition_from_state(self, src_state: str)->bool:
        '''Returns whether srcState has any labelled outgoing transition'''
        if src_state in self._transitions:
            return len(self._transitions[src_state]) > 0
        return False

    def has_proper_transition_from_state_with_symbol(self, src_state: str, symbol: str)->bool:
        '''Returns whether srcState has any outgoing transition labelled symbol.'''
        if self.has_proper_transition_from_state(src_state):
            return symbol in self._transitions[src_state].keys()
        return False

    def transitions(self)->Set[Tuple[str,str,str]]:
        '''
        Returns a set with all transitions from some state s1 to some s2
        labelled a as tuples (s1,a,s2).
        '''
        result = set()
        for src, trans in self._transitions.items():
            for symbol, states in trans.items():
                for dst in states:
                    result.add((src, symbol, dst))
        return result

    def grouped_transitions(self)->Set[Tuple[str,str,str]]:
        '''
        Returns a set with all transition from some state s1 to some state s2 a
        as tuples (s1,labels,s2), where labels is a string with all symbols for
        which there is such a transition joined with commas, including '#' if
        there is an epsilon transition from s1 to s2.
        '''
        result = set()
        trans = self.transitions()
        eps_trans = self.epsilon_transitions()
        state_pairs = {(t[0], t[2]) for t in trans}
        state_pairs.update(eps_trans)

        for p in state_pairs:
            symbols = [t[1] for t in {u for u in trans if u[0]==p[0] and u[2]==p[1]}]
            if (p[0], p[1]) in eps_trans:
                symbols.append(self._epsilonSymbol)
            # sort
            sorted_symbols = sorted(symbols)
            result.add((p[0], ','.join(sorted_symbols), p[1]))
        return result

    def add_epsilon_transition(self, src_state: str, dst_state: str):
        '''Add an epsilon transition from srcState to dstState.'''
        self.add_state(src_state)
        self.add_state(dst_state)
        if not src_state in self._epsilon_transitions:
            self._epsilon_transitions[src_state] = set()
        self._epsilon_transitions[src_state].add(dst_state)

    def epsilon_transitions(self)->Set[Tuple[str,str]]:
        '''
        Return a set with for each epsilon transition a tuple withe the source
        state and the destination state.
        '''
        result = set()
        for src, eps_trans in self._epsilon_transitions.items():
            for dst in eps_trans:
                result.add((src, dst))
        return result

    def make_initial_state(self, s: str):
        '''Make s an initial state. Assumes s is already a state of the automaton.'''
        if not s in self._states:
            raise FSAException(f"{s} is not a state of the automaton")
        self._initial_states.add(s)

    def is_initial_state(self, s: str):
        '''Check if s is an initial state. Assumes s is a state of the automaton.'''
        return s in self._initial_states

    def initial_states(self)->AbstractSet[str]:
        '''Return the (non-modifiable) set of initial states.'''
        return frozenset(self._initial_states)

    def make_final_state(self, s: str, acceptance_sets: Optional[Set[str]] = None):
        '''
        Make state s a final state. If the optional acceptanceSets is provided,
        s is added to the given acceptance sets. s is assumed to be a state of
        the automaton.
        '''
        if not s in self._states:
            raise FSAException(f"{s} is not a state of the automaton")
        if acceptance_sets is None:
            self._final_states.add(s)
        else:
            for a in acceptance_sets:
                if a=='default':
                    self._final_states.add(s)
                else:
                    if not a in self._generalized_acceptance_sets:
                        self._generalized_acceptance_sets[a] = set()
                    self._generalized_acceptance_sets[a].add(s)

    def is_final_state(self, s: str):
        '''Check if s is a final state. Assumes s is a state of the automaton.'''
        return s in self._final_states

    def generalized_acceptance_sets(self)-> Dict[str,Set[str]]:
        '''return generalized acceptance sets'''
        return copy.deepcopy(self._generalized_acceptance_sets)

    def clear_initial_states(self):
        '''Make all states non-initial.'''
        self._initial_states = set()

    def clear_final_states(self):
        '''Make all states non-final, remove all generalized acceptance sets.'''
        self._final_states = set()
        self._generalized_acceptance_sets = {}

    def final_states(self)->AbstractSet[str]:
        '''Return the (non-modifiable) set of final states.'''
        return frozenset(self._final_states)

    def make_non_final_state(self, s: str):
        '''Make s a non-final state. Assumes s is a state of the automaton.'''
        if not s in self._states:
            raise FSAException(f"{s} is not a state of the automaton")
        if s in self._final_states:
            self._final_states.remove(s)

    def accepts_with_path(self, word: str)->Tuple[bool,Optional[List[str]]]:
        """
            Check if the automaton accepts the given word (a single string of
            symbols separated by commas).
            Returns a tuple with:
            - boolean indicating if the word is accepted
            - if the previous result is True, the second element is a list of
              states with an accepting path for the give word. If the word is
              not accepted, None is returned.
        """

        # get the individual symbols from the input string
        symbols = [] if word=='' else word.split(',')

        # loop invariant: maintain set of states reachable by the symbols
        # processed so far and corresponding paths
        # initialize to the epsilon closure of the initial states.
        current_states: Set[str]
        current_paths: Dict[str,List[str]]
        current_states, current_paths = self.epsilon_closure_with_paths(self._initial_states)

        for symbol in symbols:
            current_states, paths = self.set_next_states_epsilon_closure_with_paths(current_states,
                symbol)
            new_paths: Dict[str,List[str]] = {}
            for s in current_states:
                new_paths[s] = (current_paths[paths[s][0]])[:-1] + paths[s]
            current_paths = new_paths

        reachable_final_states = current_states.intersection(self._final_states)
        res = len(reachable_final_states) != 0
        if res:
            # take arbitrary reachable state
            s = next(iter(reachable_final_states))
            return res, current_paths[s]
        return res, None

    def accepts(self, word: str)->bool:
        """
            Check if the automaton accepts the given word (a single string of
            symbols separated by commas).
        """
        res, _ = self.accepts_with_path(word)
        return res

    def is_deterministic(self)->bool:
        '''Check if the automaton is deterministic.'''
        if len(self._initial_states) > 1:
            return False
        if len(self._epsilon_transitions.keys()) > 0:
            return False
        for symbols in self._transitions.values():
            for next_states in symbols.values():
                if len(next_states) > 1:
                    return False
        return True

    def as_dfa(self)->'Automaton':
        '''Convert to a deterministic automaton.'''
        result = Automaton()

        # convert set of states to string
        def set_as_state(ss: AbstractSet[str]):
            return "{" + (",".join(sorted(ss))) + "}"

        # determine the set of reachable states
        states_explored: Set[AbstractSet[str]] = set()
        states_to_explore: Set[AbstractSet[str]] = set()
        states_to_explore.add(
            frozenset(self.epsilon_closure(self._initial_states)))

        while states_to_explore != set():
            state = states_to_explore.pop()
            states_explored.add(state)
            result.add_state(set_as_state(state))
            symbols = reduce(lambda _symbol, _state: _symbol.union(
                self.outgoing_symbols(_state)), state, set())
            for s in symbols:
                _next_state = frozenset(self.epsilon_closure(
                    self._set_next_states(state, s)))
                if not _next_state in states_explored:
                    states_to_explore.add(_next_state)
                result.add_transition(set_as_state(state), s,
                                     set_as_state(_next_state))

        # determine the initial state
        result.make_initial_state(set_as_state(
            frozenset(self.epsilon_closure(self._initial_states))))

        # determine the final states
        for s in states_explored:
            if any(t in self._final_states for t in s):
                result.make_final_state(set_as_state(s))

        return result

    def alphabet(self)->Set[str]:
        '''Return the alphabet of the automaton. I.e., all symbols that occur on transition'''
        result = set()
        for _, trans in self._transitions.items():
            result.update(trans.keys())
        return result

    def complete(self)->'Automaton':
        """Return an equivalent automaton with a total transition relation."""

        result = Automaton()

        sink_state_added = False
        sink_state = None

        alphabet = self.alphabet()
        for s in sorted(self._states):
            result.add_state(s)
            if s in self._initial_states:
                result.make_initial_state(s)
            if s in self._final_states:
                result.make_final_state(s)

            for symbol in sorted(alphabet):
                if self.has_proper_transition_from_state_with_symbol(s, symbol):
                    for t in sorted(self._transitions[s][symbol]):
                        result.add_transition(s, symbol, t)
                else:
                    if not sink_state_added:
                        sink_state_added = True
                        sink_state = result.add_state_unique("S")
                    result.add_transition(s, symbol, sink_state)  # type: ignore

        # if a new state was added it needs outgoing transitions to itself
        if sink_state_added:
            for symbol in sorted(alphabet):
                result.add_transition(sink_state, symbol, sink_state)  # type: ignore

        return result

    def complement(self)->'Automaton':
        '''Returns the complement of the automaton.'''
        # obtain a deterministic, complete automaton first
        result = self.as_dfa().complete()
        # invert the accepting set
        for s in result.states():
            if s in result.final_states():
                result.make_non_final_state(s)
            else:
                result.make_final_state(s)
        return result

    def product(self, a: 'Automaton')->'Automaton':
        '''
        Return the product automaton with automaton A. The automata
        synchronize on transitions with symbols common to their alphabets. The
        automata can independently make transitions on symbols that do not
        occur in the alphabet of the other automaton.
        '''
        result = Automaton()

        # figure out the alphabet situation
        my_alphabet = self.alphabet()
        their_alphabet = a.alphabet()
        shared_alphabet = my_alphabet.intersection(their_alphabet)
        my_private_alphabet = my_alphabet.difference(shared_alphabet)
        their_private_alphabet = their_alphabet.difference(shared_alphabet)

        def prod_state(s, t):
            return f"({s},{t})"

        # create the cartesian product states
        for s in self.states():
            for t in a.states():
                new_state = prod_state(s, t)
                result.add_state(new_state)
                if s in self._initial_states and a.is_initial_state(t):
                    result.make_initial_state(new_state)
                if s in self._final_states and a.is_final_state(t):
                    result.make_final_state(new_state)

        # create the transitions
        for s in self._states:
            for t in a.states():
                # my private alphabet transitions
                for symbol in my_private_alphabet:
                    for s_prime in self.next_states(s, symbol):
                        result.add_transition(
                            prod_state(s, t), symbol, prod_state(s_prime, t))
                # my epsilon transitions
                for s_prime in self.next_epsilon_states(s):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s_prime, t))
                # their private alphabet transitions
                for symbol in their_private_alphabet:
                    for t_prime in a.next_states(t, symbol):
                        result.add_transition(
                            prod_state(s, t), symbol, prod_state(s, t_prime))
                # their epsilon transitions
                for t_prime in a.next_epsilon_states(t):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s, t_prime))
                # our common transitions
                for symbol in shared_alphabet:
                    for s_prime in self.next_states(s, symbol):
                        for t_prime in a.next_states(t, symbol):
                            result.add_transition(
                                prod_state(s, t), symbol, prod_state(s_prime, t_prime))
        return result

    def strict_product(self, a: 'Automaton')->'Automaton':
        '''
        Return the 'strict' product automaton with automaton A. The automata
        synchronize on transitions with symbols common to their alphabets. The
        automata cannot make transitions on symbols that do not occur in the
        alphabet of the other automaton.
        '''

        result = Automaton()

        # figure out the alphabet situation
        my_alphabet = self.alphabet()
        their_alphabet = a.alphabet()
        shared_alphabet = my_alphabet.intersection(their_alphabet)

        def prod_state(s, t):
            return f"({s},{t})"

        # create the cartesian product states
        for s in self._states:
            for t in a.states():
                new_state = prod_state(s, t)
                result.add_state(new_state)
                if s in self._initial_states and a.is_initial_state(t):
                    result.make_initial_state(new_state)
                if s in self._final_states and a.is_final_state(t):
                    result.make_final_state(new_state)

        # create the transitions
        for s in self._states:
            for t in a.states():
                # my epsilon transitions
                for s_prime in self.next_epsilon_states(s):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s_prime, t))
                # her epsilon transitions
                for t_prime in a.next_epsilon_states(t):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s, t_prime))
                # our common transitions
                for symbol in shared_alphabet:
                    for s_prime in self.next_states(s, symbol):
                        for t_prime in a.next_states(t, symbol):
                            result.add_transition(
                                prod_state(s, t), symbol, prod_state(s_prime, t_prime))
        return result

    def product_buchi(self, a: 'Automaton')->'Automaton':
        '''
        Return the product Büchi automaton with Büchi automaton A. The
        automata synchronize on transitions with symbols common to their
        alphabets. The automata can independently make transitions on symbols
        that do not occur in the alphabet of the other automaton.
        '''
        result = Automaton()

        # figure out the alphabet situation
        my_alphabet = self.alphabet()
        their_alphabet = a.alphabet()
        shared_alphabet = my_alphabet.intersection(their_alphabet)
        my_private_alphabet = my_alphabet.difference(shared_alphabet)
        their_private_alphabet = their_alphabet.difference(shared_alphabet)

        def prod_state(s, t):
            return f"({s},{t})"

        # create the cartesian product states
        their_acceptance_set = set()
        for s in self._states:
            for t in a.states():
                new_state = prod_state(s, t)
                result.add_state(new_state)
                if s in self._initial_states and a.is_initial_state(t):
                    result.make_initial_state(new_state)
                # take the acceptance conditions from self
                # record the acceptance conditions from her to be added later
                if s in self._final_states:
                    result.make_final_state(new_state)
                if a.is_final_state(t):
                    their_acceptance_set.add(new_state)

        # create the transitions
        for s in self._states:
            for t in a.states():
                # my private alphabet transitions
                for symbol in my_private_alphabet:
                    for s_prime in self.next_states(s, symbol):
                        result.add_transition(
                            prod_state(s, t), symbol, prod_state(s_prime, t))
                # my epsilon transitions
                for s_prime in self.next_epsilon_states(s):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s_prime, t))
                # their private alphabet transitions
                for symbol in their_private_alphabet:
                    for t_prime in a.next_states(t, symbol):
                        result.add_transition(
                            prod_state(s, t), symbol, prod_state(s, t_prime))
                # their epsilon transitions
                for t_prime in a.next_epsilon_states(t):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s, t_prime))
                # our common transitions
                for symbol in shared_alphabet:
                    for s_prime in self.next_states(s, symbol):
                        for t_prime in a.next_states(t, symbol):
                            result.add_transition(
                                prod_state(s, t), symbol, prod_state(s_prime, t_prime))
        return result.add_generalized_buchi_acceptance_sets(set([frozenset(their_acceptance_set)]))

    def strict_product_buchi(self, a: 'Automaton')->'Automaton':
        '''
        Return the 'strict' product Büchi automaton with büchi automaton A.
        The automata synchronize on transitions with symbols common to their
        alphabets. The automata cannot make transitions on symbols that do not
        occur in the alphabet of the other automaton.
        '''
        result = Automaton()

        # figure out the alphabet situation
        my_alphabet = self.alphabet()
        their_alphabet = a.alphabet()
        shared_alphabet = my_alphabet.intersection(their_alphabet)

        def prod_state(s, t):
            return f"({s},{t})"

        # create the cartesian product states
        # herAcceptanceSet = set()
        for s in self._states:
            for t in a.states():
                new_state = prod_state(s, t)
                result.add_state(new_state)
                if s in self._initial_states and a.is_initial_state(t):
                    result.make_initial_state(new_state)
                # determine the generalized acceptance sets
                acceptance_sets = set()
                if s in self._final_states:
                    acceptance_sets.add("A")
                if a.is_final_state(t):
                    acceptance_sets.add("B")
                for acc_set, acc_set_states in self._generalized_acceptance_sets.items():
                    if s in acc_set_states:
                        acceptance_sets.add("A_" + acc_set)
                for acc_set, acc_set_states in a.generalized_acceptance_sets().items():
                    if s in acc_set_states:
                        acceptance_sets.add("B_" + acc_set)
                result.make_final_state(new_state, acceptance_sets)

        # create the transitions
        for s in self._states:
            for t in a.states():
                # my epsilon transitions
                for s_prime in self.next_epsilon_states(s):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s_prime, t))
                # her epsilon transitions
                for t_prime in a.next_epsilon_states(t):
                    result.add_epsilon_transition(
                        prod_state(s, t), prod_state(s, t_prime))
                # our common transitions
                for symbol in shared_alphabet:
                    for s_prime in self.next_states(s, symbol):
                        for t_prime in a.next_states(t, symbol):
                            result.add_transition(
                                prod_state(s, t), symbol, prod_state(s_prime, t_prime))

        return result

    def language_empty(self)->Tuple[bool,Optional[List[str]],Optional[List[str]]]:
        '''
        Checks if the FSA language is empty. Returns in addition to a Boolean,
        an accepted word and path if the language is not empty.
        '''

        # explore if a final state is reachable from an initial state

        # check if one of the initial states is final
        for s in sorted(self._initial_states):
            if s in self._final_states:
                return (False, [], [s])

        # non-final states that remain to be explored
        states_to_explore = sorted(list(self._initial_states))
        # invariant: states that have already been explored, should all be keys in backTrack
        states_explored: Set[str] = set()
        # keep track of incoming symbol and state
        back_track: Dict[str,Tuple[str,str]] = {}
        while len(states_to_explore) > 0:
            state = states_to_explore.pop(0)
            states_explored.add(state)
            # for all epsilon transitions
            for s in sorted(self.next_epsilon_states(state)):
                if not s in states_explored:
                    states_to_explore.append(s)
                    back_track[s] = (self._epsilonSymbol, state)
                    if s in self._final_states:
                        word, path = self._trace_accepting_word_and_path(s, back_track)
                        return False, word, path

            # for all symbol transitions
            for symbol in sorted(self.outgoing_symbols(state)):
                for s in sorted(self.next_states(state, symbol)):
                    if not s in states_explored:
                        states_to_explore.append(s)
                        back_track[s] = (symbol, state)
                    if s in self._final_states:
                        word, path = self._trace_accepting_word_and_path(s, back_track)
                        return False, word, path
        # no final state was reached
        return (True, None, None)

    def language_empty_buchi(self)-> \
        Tuple[bool,Optional[List[str]],Optional[List[str]],Optional[List[str]],Optional[List[str]]]:
        '''
        Checks if the Buchi language is empty. Returns an accepted word
        9prefix, repetition) and path (prefix, repetition) if the language is
        not empty.
        '''

        # Step 1: find all reachable accepting states and determine a word sigma that leads to it

        # Step 2: For each accepting state s do:
        #   - find all states that can be reached with one non-epsilon symbol
        #     and remember that symbol a (needed to ensure the final result is
        #     an infinite word)
        #   - check if s can be reached from any of those states by a word tau
        #   - (TODO: it may be OK to stop searching when we reaching any of the
        #     accepting states we already covered)
        #   - if yes, the automaton accepts the word (sigma)(a tau)**

        # Step 1, determine the reachable accepting states
        reachable_states: Set[str]
        words: Dict[str,List[str]]
        paths: Dict[str,List[str]]
        reachable_states, words, paths = self.reachable_states_with_words_and_paths()
        reachable_final_states: Set[str] = reachable_states.intersection(self._final_states)

        # Step 2, check for each reachable accepting state if there is a cycle
        # with a non-empty word that returns to it
        for s in sorted(reachable_final_states):
            # to ensure that the cycle word is non-empty, first determine the
            # states reachable from s with a non-epsilon symbol, and remember
            # that symbol for each state
            s_closure = self.epsilon_closure(set([s]))
            final_plus_symbol_reachable_states: Set[str] = set()
            single_symbol: Dict[str,List[str]] = {}
            for t in sorted(s_closure):
                for symbol in sorted(self.outgoing_symbols(t)):
                    states = self.epsilon_closure(self.next_states(t, symbol))
                    final_plus_symbol_reachable_states.update(states)
                    for u in sorted(states):
                        single_symbol[u] = [symbol]

            # test if s is reachable from any state in finalPlusSymbolReachableStates
            cycle_reachable_states, cycle_words, cycle_paths = \
                self.reachable_states_with_words_and_paths(final_plus_symbol_reachable_states,
                                                      single_symbol)
            if s in cycle_reachable_states:
                return (False, words[s], cycle_words[s], paths[s], cycle_paths[s])

        # No Cycle found
        return (True, None, None, None, None)

    def language_included(self, a: 'Automaton')->Tuple[bool,Optional[List[str]]]:
        '''
        Check if the language of the automaton is included in the language of automaton A.
        If not, a word is returned that is in the language of the automaton,
        but not in the language of A
        '''
        a_c = a.complement()
        p = a_c.strict_product(self)
        bool_result, word, _ = p.language_empty()
        return (bool_result, word)

    def sub_automaton(self, states: Set[str])->'Automaton':
        ''' return a sub-automaton containing only the states in the set states '''

        result = Automaton()
        # make states
        for s in self._states.intersection(states):
            result.add_state(s)
            if s in self._initial_states:
                result.make_initial_state(s)
            if s in self._final_states:
                result.make_final_state(s)

        # make epsilon transitions
        for s, e_trans in self._epsilon_transitions.items():
            if s in states:
                for t in e_trans:
                    if t in states:
                        result.add_epsilon_transition(s, t)

        # make regular transitions
        for s, trans in self._transitions.items():
            if s in states:
                for symbol in trans:
                    for t in trans[symbol]:
                        if t in states:
                            result.add_transition(s, symbol, t)

        return result

    def eliminate_reachability(self)-> 'Automaton':
        '''
        Reduce the size of the automaton by removing unreachable states and
        states from which no final state is reachable.
        '''

        # remove unreachable states
        states = self.reachable_states().intersection(self.reachable_states_final())
        return self.sub_automaton(states)

    def eliminate_states_without_outgoing_transitions(self)-> 'Automaton':
        '''
        Return an automaton where all states without outgoing transitions
        are removed. For Büchi automata this results in an equivalent automaton.
        '''
        to_eliminate: Set[str] = set()
        for s in self._states:
            if len(self.outgoing_symbols_with_epsilon(s)) ==0:
                to_eliminate.add(s)

        if len(to_eliminate) ==0:
            return self

        return self.sub_automaton(self._states.difference(to_eliminate)). \
            eliminate_states_without_outgoing_transitions()

    def partition_refinement(self)->Tuple[Set[AbstractSet[str]],Dict[str,AbstractSet[str]]]:
        '''Return equivalence classes according to a partition refinement process.'''

        def _create_partition(partitions: Set[AbstractSet[str]], \
                              partition_map: Dict[str,AbstractSet[str]], set_of_states: Set[str]):
            '''
            Create a partition for setOfStates, add it to partitions and
            update the partition map accordingly.
            '''
            f_set_of_states = frozenset(set_of_states)
            partitions.add(f_set_of_states)
            for s in set_of_states:
                partition_map[s] = f_set_of_states

        def _partition_refinement_edges_equivalent(s1: str, s2: str)->bool:
            '''Check if states s1 and s2 are considered equivalent.'''

            # s1 and s2 are equivalent if for every s1-a->C, s2-a->C and vice versa
            labels: Set[str] = set()
            labels.update(self.outgoing_symbols_set(self.epsilon_closure(set([s1]))))
            labels.update(self.outgoing_symbols_set(self.epsilon_closure(set([s2]))))

            ecs1 = self.epsilon_closure(set([s1]))
            ecs2 = self.epsilon_closure(set([s2]))

            # for every label, compare outgoing edges
            for l in labels:
                # collect classes of states in ns1 and ns2
                cs1: Set[AbstractSet[str]] = set()
                cs2: Set[AbstractSet[str]] = set()
                for t in ecs1:
                    for s in self.epsilon_closure(self.next_states(t, l)):
                        cs1.add(partition_map[s])
                for t in ecs2:
                    for s in self.epsilon_closure(self.next_states(t, l)):
                        cs2.add(partition_map[s])
                # compare classes
                if cs1 != cs2:
                    return False

            return True

        # make initial partition on states that agree on final-ness
        partitions: Set[AbstractSet[str]] = set()
        # final are states from which a final state is reachable with epsilon moves
        states_f = self.backward_epsilon_closure(self._final_states)
        # non-final, others
        states_nf = self._states.difference(states_f)
        p_f = None
        if states_f:
            p_f = frozenset(states_f)
            partitions.add(p_f)
        p_nf = None
        if len(states_nf):
            p_nf = frozenset(states_nf)
            partitions.add(p_nf)

        partition_map: Dict[str,AbstractSet[str]] = {}
        for s in states_f:
            partition_map[s] = p_f  # type: ignore I'm sure p_f is not None here
        for s in states_nf:
            partition_map[s] = p_nf  # type: ignore I'm sure p_nf is not None here

        old_partitions: Set[AbstractSet[str]] = set()

        while len(old_partitions) != len(partitions):
            # print(partitions)
            new_partitions: Set[AbstractSet[str]] = set()
            for e_class in partitions:
                # pick arbitrary state from class
                s1 = next(iter(e_class))

                equiv_set: Set[str] = set()
                remaining_set: Set[str] = set()
                equiv_set.add(s1)

                # check whether all other states can go with the same labels to
                # the same set of other equivalence classes.
                for s2 in e_class:
                    if s2 != s1:
                        if _partition_refinement_edges_equivalent(s1, s2):
                            equiv_set.add(s2)
                        else:
                            remaining_set.add(s2)

                # if not, split the class
                if len(equiv_set) == len(e_class):
                    _create_partition(new_partitions, partition_map, equiv_set)
                else:
                    _create_partition(new_partitions, partition_map, equiv_set)
                    _create_partition(new_partitions, partition_map, remaining_set)

            old_partitions = partitions
            partitions = new_partitions

        return partitions, partition_map

    def minimize(self)->'Automaton':
        '''Implements a partition refinement strategy to reduce the size of the FSA.'''

        def set_as_state(ss):
            return "{" + (",".join(sorted(ss))) + "}"

        # remove unreachable states
        # remove states from which final states are not reachable
        interim = self.eliminate_reachability()
        # find equivalent states through partition refinement.
        partitions, partition_map = interim.partition_refinement()

        result = Automaton()

        # make states
        for p in partitions:
            ns = set_as_state(p)
            result.add_state(ns)
            s = next(iter(p))
            if s in interim.backward_epsilon_closure(set(interim.final_states())):
                result.make_final_state(ns)

            # determine initial states
            # a partition is initial if one of its states was initial
            for s in p:
                if s in interim.initial_states():
                    result.make_initial_state(ns)

        # make transitions
        for p in partitions:
            # take a representative state
            s = next(iter(p))
            for t in interim.epsilon_closure(set([s])):
                for symbol in interim.outgoing_symbols(t):
                    for u in interim.next_states(t, symbol):
                        result.add_transition(set_as_state(partition_map[s]), \
                                              symbol, set_as_state(partition_map[u]))
                if t in interim._epsilon_transitions:
                    for u in interim._epsilon_transitions[t]:
                        if partition_map[s] != partition_map[u]:
                            result.add_epsilon_transition(set_as_state(partition_map[s]), \
                                set_as_state(partition_map[u]))
        return result

    def minimize_buchi(self)->'Automaton':
        '''Implements a partition refinement strategy to reduce the size of the Büchi automaton.'''

        # eliminate states from which not all acceptance sets are reachable
        interim = self.eliminate_states_without_outgoing_transitions()
        return  interim.minimize()

    def states_in_bfs_order(self)->List[str]:
        '''Return a list of state in breadth-first order'''
        result = []
        self._breadth_first_search(result.append)
        return result

    def relabel_states(self)->'Automaton':
        '''
        Return the automaton with states relabeled 'S' with a number in a
        breadth first manner.
        '''

        def _state_name(n:int)->str:
            return f"S{n}"

        def _create_state(s: str):
            nonlocal n
            new_state = _state_name(n)
            state_dict[s] = new_state
            result.add_state(new_state)
            n += 1

        result = Automaton()
        state_dict = {}
        n = 1

        self._breadth_first_search(_create_state)

        for i in self._initial_states:
            result.make_initial_state(state_dict[i])

        for f in self._final_states:
            result.make_final_state(state_dict[f])

        if self.has_generalized_acceptance_sets():
            for gas, acceptance_set in self._generalized_acceptance_sets.items():
                result.add_generalized_buchi_acceptance_set(gas, \
                        set(map(lambda s: state_dict[s], acceptance_set)))

        for s, e_trans in self._epsilon_transitions.items():
            for t in e_trans:
                result.add_epsilon_transition(state_dict[s], state_dict[t])

        for s, trans in self._transitions.items():
            for symbol, s_trans in trans.items():
                for t in s_trans:
                    result.add_transition(state_dict[s], symbol, state_dict[t])
        return result

    def eliminate_epsilon_transitions(self)->'Automaton':
        '''Eliminate epsilon transitions from the automaton.'''

        result = Automaton()

        for s in self._states:
            result.add_state(s)

        for s, trans in self._transitions.items():
            for symbol, s_trans in trans.items():
                for t in s_trans:
                    # for u in self.epsilonClosure(set([t])):
                    for v in self.backward_epsilon_closure(set([s])):
                        result.add_transition(v, symbol, t)

        for s in self._states:
            if s in self._initial_states:
                result.make_initial_state(s)
                # for t in self.epsilonClosure(set([s])):
                #     result.makeInitialState(t)
            if s in self._final_states:
                for t in self.backward_epsilon_closure(set([s])):
                    result.make_final_state(t)
        return result


    def _as_dsl_symbol(self, symbol: str)->str:
        '''Escape quotes.'''
        if re.match(r"[a-zA-Z][a-zA-Z0-9]*", symbol):
            return symbol
        return '"' + symbol.replace('"', '\\"') + '"'

    def as_dsl(self, name: str)->str:
        '''Return a string representing the automaton in the domain-specific language,'''

        def _add_state_with_attributes(u: str, states_output:Set[str], output: StringIO):
            output.write(u)
            if not u in states_output:
                # if it is the first time we are using this state, add its attributes
                self._dsl_output_state_attributes(u, output)
                states_output.add(u)

        # keep track of the states that have been output
        states_output: Set[str] = set()
        # create string writer for the output
        output = StringIO()
        # write header
        output.write(f"finite state automaton {name} {{\n")
        # for all transitions (collecting multiple transitions into one)
        for (s, symbols, t) in sorted(self._dsl_multi_symbol_transitions()):
            # write the transition
            output.write("\t")
            _add_state_with_attributes(s, states_output, output)
            if len(symbols)==1 and self._epsilonSymbol in symbols:
                output.write(" ----> ")
            else:
                output.write(" -- ")
                output.write(", ".join([self._as_dsl_symbol(symbol) for symbol in sorted(symbols)]))
                output.write(" --> ")
            _add_state_with_attributes(t, states_output, output)
            output.write("\n")
        # write the remaining states without transitions
        remaining_states = self._states.difference(states_output)
        if len(remaining_states):
            output.write("\tstates\n")
            for s in remaining_states:
                output.write("\t")
                _add_state_with_attributes(s, states_output, output)
                output.write("\n")
        output.write("}\n")
        result = output.getvalue()
        output.close()
        return result

    @staticmethod
    def _parsing_add_state_with_labels(fsa: 'Automaton', s: str, \
                                       labels:Set[str], acceptance_sets:Set[str]):
        '''Add a state with the given attributes.'''
        fsa.add_state(s)
        for attr in labels:
            if attr in ('initial', 'i'):
                fsa.make_initial_state(s)
            if attr in ('final', 'f'):
                fsa.make_final_state(s, acceptance_sets)

    @staticmethod
    def from_dsl(dsl_string)->Tuple[str,'Automaton']:
        '''Create an automaton from the DSL string.'''

        factory = {}
        factory['Init'] = Automaton
        factory['addTransitionPossiblyEpsilon'] = lambda fsa, s, t, symbol: \
            fsa.addEpsilonTransition(s, t) if symbol == \
                Automaton._epsilonSymbol else fsa.addTransition(s, symbol, t)
        factory['AddEpsilonTransition'] = lambda fsa, s, t : fsa.addEpsilonTransition(s, t)
        factory['AddState'] = Automaton._parsing_add_state_with_labels
        name, fsa = parse_fsa_dsl(dsl_string, factory)
        if name is None or fsa is None:
            sys.exit(1)
        return name, fsa

    def reachable_states(self)->Set[str]:
        ''' return a set of all states reachable from an initial state '''
        result, _, _ = self.reachable_states_with_words_and_paths()
        return result

    def reachable_states_with_words_and_paths(self, \
            starting_states:Optional[Iterable[str]]=None, \
            starting_words:Optional[Dict[str,List[str]]]=None, \
            starting_paths=None)->Tuple[Set[str],Dict[str,List[str]],Dict[str,List[str]]]:
        '''
        return a set of all states reachable from any state in startingStates
        and for each a word and a path by which is is reached. If
        startingStates is omitted, the initial states are used
        '''
        if starting_states is None:
            starting_states = sorted(self._initial_states)
        result = set()
        words: Dict[str,List[str]]
        if starting_words is None:
            words = {}
            for  s in starting_states:
                words[s] = []
        else:
            words = starting_words
        if starting_paths is None:
            paths = {}
            for  s in starting_states:
                paths[s] = [s]
        else:
            paths = starting_paths
        states_to_explore = sorted(list(starting_states))
        while len(states_to_explore) > 0:
            s = states_to_explore.pop()
            result.add(s)
            if s in self._epsilon_transitions:
                for t in sorted(self._epsilon_transitions[s]):
                    if not t in result:
                        states_to_explore.append(t)
                        words[t] = words[s]
                        paths[t] = paths[s] +[t]
            if s in self._transitions:
                for symbol in sorted(self._transitions[s]):
                    for t in sorted(self._transitions[s][symbol]):
                        if not t in result:
                            states_to_explore.append(t)
                            words[t] = words[s] + [symbol]
                            paths[t] = paths[s] + [t]

        return result, words, paths

    def reachable_states_final(self)->Set[str]:
        ''' return the set of all states from which a final state is reachable '''
        result: Set[str] = set()
        states_to_explore = set(self._final_states)
        while states_to_explore != set():
            s = states_to_explore.pop()
            result.add(s)
            # check epsilon transitions
            for t, trans in self._epsilon_transitions.items():
                for u in trans:
                    if u == s:
                        if not t in result:
                            states_to_explore.add(t)
            # check regular transitions
            for t, trans in self._transitions.items():
                for symbol in trans:
                    for u in trans[symbol]:
                        if u == s:
                            if not t in result:
                                states_to_explore.add(t)
        return result

    def add_generalized_buchi_acceptance_set(self, n: str, a:AbstractSet[str]):
        '''
        add a generalized acceptance set A named N
        '''
        self._generalized_acceptance_sets[n] = set(a)

    def add_generalized_buchi_acceptance_sets(self, a:Iterable[AbstractSet[str]])->'Automaton':
        '''
        return a new non-generalized Büchi automaton with the added generalized
        acceptance sets incorporated
        '''
        res, _ = self.add_generalized_buchi_acceptance_sets_with_state_map(a)
        return res

    def add_generalized_buchi_acceptance_sets_with_state_map(self, \
            a:Iterable[AbstractSet[str]])->Tuple['Automaton',Dict[str,str]]:
        '''
        return a new non-generalized Büchi automaton with the added generalized
        acceptance sets incorporated and a map linking the new states to the
        original states
        '''

        def _new_state(s: str, n: int)->str:
            return f"({s},F{str(n)})"

        state_map:Dict[str,str] = {}

        # create a copy of every state for every acceptance set.
        # label final state accordingly
        # add transitions to state in same layer for non-accepting source states
        # or state in next layer if it is accepting
        res = Automaton()
        acceptance_sets:List[AbstractSet[str]] = []
        if len(self._final_states) > 0:
            acceptance_sets.append(self._final_states)
        for aa in a:
            acceptance_sets.append(aa)
        n = len(acceptance_sets)

        # create states
        for n in range(n):
            for s in self._states:
                ns = _new_state(s,n)
                state_map[ns] = s
                res.add_state(ns)

        # set initial states
        for s in self._initial_states:
            res.make_initial_state(_new_state(s,0))

        # set final state s
        for n in range(n):
            for s in acceptance_sets[n]:
                res.make_final_state(_new_state(s,n))

        # add transitions
        for n in range(n):
            nxt = (n+1)%n
            for s, trans in self._transitions.items():
                for symbol, s_trans in trans.items():
                    for t in s_trans:
                        if s in acceptance_sets[n]:
                            res.add_transition(_new_state(s,n), symbol, _new_state(t,nxt))
                        else:
                            res.add_transition(_new_state(s,n), symbol, _new_state(t,n))
            for s, s_trans in self._epsilon_transitions.items():
                for t in s_trans:
                    if s in acceptance_sets[n]:
                        res.add_epsilon_transition(_new_state(s,n), _new_state(t,nxt))
                    else:
                        res.add_epsilon_transition(_new_state(s,n), _new_state(t,n))

        return res, state_map

    def has_generalized_acceptance_sets(self)->bool:
        '''Test if the automaton has generalized acceptance sets.'''
        return len(self._generalized_acceptance_sets) > 0


    def as_regular_buchi_automaton(self)->'Automaton':
        '''
        Convert to an equivalent regular Büchi automaton, i.e., without
        generalized acceptance sets.
        '''
        res, _ = self.as_regular_buchi_automaton_with_state_map()
        return res

    def as_regular_buchi_automaton_with_state_map(self)->Tuple['Automaton',Dict[str,str]]:
        '''
        Convert to an equivalent regular Büchi automaton, i.e., without
        generalized acceptance sets. Return automaton and a map relating the
        states in the new automaton to states in the old automaton.
        '''
        if len(self._generalized_acceptance_sets) == 0:
            state_map = {}
            for s in self._states:
                state_map[s] = s
            return self, state_map
        return self.add_generalized_buchi_acceptance_sets_with_state_map(\
            list(self._generalized_acceptance_sets.values()))

    def _dsl_multi_symbol_transitions(self)->Set[Tuple[str,Tuple[str],str]]:
        '''collect common transitions into multi-labels, including epsilon transitions'''
        reorg = {}

		# handle all regular transitions
        for s, trans in self._transitions.items():
			# ensure s is a key in reorg
            if not s in reorg:
                reorg[s] = {}

			# for all transitions from s collect the symbols going to states t
            for symbol, s_trans in trans.items():
                for t in s_trans:
					# ensure that t is a key in reorg[s]
                    if not t in reorg[s]:
                        reorg[s][t] = set()
					# add the symbol to the set
                    reorg[s][t].add(symbol)

		# handle all epsilon transitions
        for s, eps_s in self._epsilon_transitions.items():
			# ensure s is a key in reorg
            if not s in reorg:
                reorg[s] = {}
            for t in eps_s:
				# ensure that t is a key in reorg[s]
                if not t in reorg[s]:
                    reorg[s][t] = set()
				# add the symbol to the set
                reorg[s][t].add(self._epsilonSymbol)

		# create the results
        result: Set[Tuple[str,Tuple[str],str]] = set()
        for s, ss in reorg.items():
            for t in ss:
                result.add((s, tuple(sorted(ss[t])), t)) # type: ignore
        return result

    def _dsl_output_state_attributes(self, state: str, output: StringIO):
        '''Output state attributes to output'''
        if state in self._initial_states:
            output.write(" initial")
            if state in self._final_states:
                output.write("; final")
            return
        generalized_sets = {a for a, a_s in self._generalized_acceptance_sets.items() if \
             state in a_s}
        if state in self._final_states or len(generalized_sets)>0:
            output.write(" final")
        if len(generalized_sets)>0:
            if state in self._final_states:
                generalized_sets.add('default')
            output.write(f" [{", ".join(generalized_sets)}]")

    def outgoing_symbols(self, state: str)->AbstractSet[str]:
        '''Return the set of outgoing symbols from state.'''
        if not state in self._transitions:
            return set()
        return frozenset(self._transitions[state].keys())


    def outgoing_symbols_set(self, set_of_states: Set[str])->AbstractSet[str]:
        '''Return the set of outgoing symbols from any state from setOfStates.'''
        res = set()
        for s in set_of_states:
            res.update(self.outgoing_symbols(s))
        return frozenset(res)

    def outgoing_symbols_with_epsilon(self, state: str)->AbstractSet[str]:
        '''Return the set of outgoing symbols from state, including # for epsilon transitions.'''
        result = set(self.outgoing_symbols(state))
        if state in self._epsilon_transitions:
            result.add(Automaton._epsilonSymbol)
        return frozenset(result)

    def next_states(self, state: str, symbol: str)->Set[str]:
        """
        Return the set of states reachable from 'state' by a transition
        labelled 'symbol', where symbol is a non-epsilon symbol.
        """
        if not state in self._transitions:
            return set()
        if not symbol in self._transitions[state]:
            return set()
        return self._transitions[state][symbol]

    def next_states_with_epsilon(self, state: str, symbol: str)->Set[str]:
        """
        Return the set of states reachable from 'state' by a transition
        labelled 'symbol', where symbol can be an epsilon symbol.
        """
        if symbol == Automaton._epsilonSymbol:
            return self._epsilon_transitions[state]
        if not state in self._transitions:
            return set()
        if not symbol in self._transitions[state]:
            return set()
        return self._transitions[state][symbol]

    def next_epsilon_states(self, state: str)->Set[str]:
        """
        Return the set of states reachable from 'state' by a single epsilon transition.
        """
        if not state in self._epsilon_transitions:
            return set()
        return self._epsilon_transitions[state]

    def next_states_epsilon_closure_with_paths(self, state: str, symbol: str)-> \
        Tuple[Set[str],Dict[str,List[str]]]:
        """
        Return the set of states reachable from 'state' by a sequence of
        transitions including an arbitrary number of epsilon transitions and
        one 'symbol' transition, where symbol is not epsilon.
        Returns a tuple with:
        - a set of states thus reachable
        - a dictionary with for every reachable state s, a path starting in
          'state' and ending in s.
        """
        return self.set_next_states_epsilon_closure_with_paths(set([state]), symbol)

    def set_next_states_epsilon_closure_with_paths(self, states: Set[str], symbol: str) -> \
        Tuple[Set[str],Dict[str,List[str]]]:
        """
        Return the set of states reachable from a state from 'states' by a
        sequence of transitions including an arbitrary number of epsilon
        transitions and one 'symbol' transition, where symbol is not epsilon.
        Returns a tuple with:
        - a set of states thus reachable
        - a dictionary with for every reachable state s, a path starting in a
          state from 'states' and ending in s.
        """
        pre_epsilon_reachable_states, pre_paths = self.epsilon_closure_with_paths(states)
        after_symbol_states: Set[str] = set()
        after_symbol_paths: Dict[str,List[str]] = {}
        for s in pre_epsilon_reachable_states:
            for t in self.next_states(s, symbol):
                after_symbol_states.add(t)
                after_symbol_paths[t] = pre_paths[s] + [t]
        post_epsilon_reachable_states, post_paths = \
            self.epsilon_closure_with_paths(after_symbol_states)
        res_paths: Dict[str,List[str]] = {}
        for s in post_epsilon_reachable_states:
            res_paths[s] = (after_symbol_paths[post_paths[s][0]])[:-1] + post_paths[s]

        return post_epsilon_reachable_states, res_paths


    def _set_next_states(self, states: AbstractSet[str], symbol: str)->Set[str]:
        '''Return the set of states reachable from a state in states with a symbol transition.'''
        n_states = set()
        for s in states:
            n_states.update(self.next_states(s, symbol))
        return n_states

    def add_state_unique(self, state: str)->str:
        '''Add a state named state, but ensure that it is unique by potentially
        modifying the name.'''
        if not state in self._states:
            self.add_state(state)
            return state
        n = 1
        while state+str(n) in self._states:
            n += 1
        new_state = state+str(n)
        self.add_state(new_state)
        return new_state

    def epsilon_closure_with_paths(self, set_of_states: Set[str])-> \
        Tuple[Set[str],Dict[str,List[str]]]:
        """
        Determine the epsilon closure of the given set of states. Return a tuple with:
        - the set of states reachable by zero or more epsilon transitions
        - a dictionary that maps each of the reachable states s to a path
            of states starting from one of the states in the initial set and
            ending in s.
        """
        res = set_of_states
        paths = {}
        for s in set_of_states:
            paths[s] = [s]
        n = 0
        while n < len(res):
            n = len(res)
            new_res = res.copy()
            for s in res:
                if s in self._epsilon_transitions:
                    new_states = self._epsilon_transitions[s]
                    new_res.update(new_states)
                    for t in new_states:
                        if t not in paths:
                            paths[t] = paths[s] + [t]
            res = new_res
        return res, paths

    def epsilon_closure(self, set_of_states: Set[str])->Set[str]:
        '''Determine the set of states reachable by epsilon transitions from
        any state in setOfStates.'''
        res, _ = self.epsilon_closure_with_paths(set_of_states)
        return res

    def backward_epsilon_closure(self, set_of_states: Set[str])->Set[str]:
        """
        Determine the backward epsilon closure of the given set of states.
        Return a set of states reachable by zero or more epsilon transitions
        taken backward.
        """
        res = set_of_states
        n = 0
        while n < len(res):
            n = len(res)
            new_res = res.copy()
            for s in res:
                for t, eps_states in self._epsilon_transitions.items():
                    if s in eps_states:
                        new_res.add(t)
            res = new_res
        return res

    def _trace_accepting_word_and_path(self, s: str, back_track: \
                                       Dict[str,Tuple[str,str]])->Tuple[List[str],List[str]]:
        '''
        Reconstruct accepting word and path from state s tracking back to an
        initials tate using backTrack. Return bool for success and if
        successful word and path.
        '''
        word = []
        path = [s]
        t = s
        while not t in self._initial_states:
            (sym, t) = back_track[t]
            if sym != self._epsilonSymbol:
                word.insert(0, sym)
            path.insert(0, t)
        return (word, path)

    def _successor_states(self, s:str)->Set[str]:
        '''Return a set of successor states of s by any transition or epsilon transition.'''
        result = set()
        if s in self._transitions:
            for symbol in self._transitions[s]:
                result.update(self.next_states(s, symbol))
        if s in self._epsilon_transitions:
            result.update(self._epsilon_transitions[s])
        return result

    def _breadth_first_search(self, visit: Callable[[str],None]):
        visited_states: Set[str] = set()
        states_to_visit = sorted(list(self._initial_states))
        set_of_states_to_visit = set(self._initial_states)
        while len(states_to_visit) > 0:
            s = states_to_visit.pop(0)
            set_of_states_to_visit.remove(s)
            visit(s)
            visited_states.add(s)
            new_states = self._successor_states(s).difference(visited_states). \
                difference(set_of_states_to_visit)
            for t in sorted(new_states):
                states_to_visit.append(t)
                set_of_states_to_visit.add(t)

        # remaining states in no particular order
        for s in self._states.difference(visited_states):
            visit(s)

    def __str__(self)->str:
        return f"({self._states}, {self._initial_states}, {self.\
            _final_states}, {self._transitions})"
