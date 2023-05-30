from functools import reduce
from io import StringIO
import re
from typing import AbstractSet, Callable, Dict, Iterable, List, Optional,Set, Tuple
from finitestateautomata.libfsagrammar import parseFSADSL


class Automaton(object):

    _epsilonSymbol:str = '#'

    # set of states
    _states: Set[str]
    # map from states to map from symbol to set of next states
    _transitions: Dict[str,Dict[str,Set[str]]]
    # map from states to set of epsilon next states
    _epsilonTransitions: Dict[str,Set[str]]
    # set of initial states
    _initialStates: Set[str]
    # set of final states
    _finalStates: Set[str]
    # map from a name to a set of states forming an acceptance set
    # note the the set of final states acts as one of the generalized acceptance sets
    _generalizedAcceptanceSets: Dict[str,Set[str]]

    def __init__(self):
        self._states = set()
        self._transitions = dict()
        self._epsilonTransitions = dict()
        self._initialStates = set()
        self._finalStates = set()
        self._generalizedAcceptanceSets = dict()
    
    def addState(self, s: str):
        '''Add a state.'''
        self._states.add(s)

    def addStates(self, setOfStates: Set[str]):
        '''Add a set of states.'''
        self._states.update(setOfStates)

    def states(self)->AbstractSet[str]:
        '''Returns the (non-modifiable) set of states.'''
        return frozenset(self._states)

    def addTransition(self, srcState: str, symbol: str, dstState: str):
        '''Add a transition from srcState to dstState, labelled with symbol.'''
        self.addState(srcState)
        self.addState(dstState)
        # update the transitions
        if not srcState in self._transitions.keys():
            self._transitions[srcState] = dict()
        if not symbol in self._transitions[srcState].keys():
            self._transitions[srcState][symbol] = set()
        self._transitions[srcState][symbol].add(dstState)

    def hasProperTransitionFromState(self, srcState: str)->bool:
        '''Returns whether srcState has any labelled outgoing transition'''
        if srcState in self._transitions.keys():
            return len(self._transitions[srcState]) > 0
        else:
            return False

    def hasProperTransitionFromStateWithSymbol(self, srcState: str, symbol: str)->bool:
        '''Returns whether srcState has any outgoing transition labelled symbol.'''
        if self.hasProperTransitionFromState(srcState):
            return symbol in self._transitions[srcState].keys()
        else:
            return False

    def transitions(self)->Set[Tuple[str,str,str]]:
        '''Returns a set with all transition from some state s1 to some s2 labelled a as tuples (s1,a,s2).'''
        result = set()
        for src in self._transitions:
            for symbol in self._transitions[src]:
                for dst in self._transitions[src][symbol]:
                    result.add((src, symbol, dst))
        return result

    def groupedTransitions(self)->Set[Tuple[str,str,str]]:
        '''Returns a set with all transition from some state s1 to some state s2 a as tuples (s1,labels,s2), where labels is a string with all symbols for which there is such a transition joined with commas, including '#' if there is an epsilon transition from s1 to s2.'''
        result = set()
        trans = self.transitions()
        epsTrans = self.epsilonTransitions()
        statePairs = set([(t[0], t[2]) for t in trans])
        statePairs.update(epsTrans)

        for p in statePairs:
            symbols = [t[1] for t in {u for u in trans if u[0]==p[0] and u[2]==p[1]}]
            if (p[0], p[1]) in epsTrans:
                symbols.append(self._epsilonSymbol)
            # sort
            sortedSymbols = sorted(symbols)
            result.add((p[0], ','.join(sortedSymbols), p[1]))
        return result

    def addEpsilonTransition(self, srcState: str, dstState: str):
        '''Add an epsilon transition from srcState to dstState.'''
        self.addState(srcState)
        self.addState(dstState)
        if not srcState in self._epsilonTransitions.keys():
            self._epsilonTransitions[srcState] = set()
        self._epsilonTransitions[srcState].add(dstState)

    def epsilonTransitions(self)->Set[Tuple[str,str]]:
        '''Return a set with for each epsilon transition a tuple withe the source state and the destination state.'''
        result = set()
        for src in self._epsilonTransitions:
            for dst in self._epsilonTransitions[src]:
                result.add((src, dst))
        return result

    def makeInitialState(self, s: str):
        '''Make s an initial state. Assumes s is already a state of the automaton.'''
        if not s in self._states:
            raise Exception("{} is not a state of the automaton".format(s))
        self._initialStates.add(s)

    def initialStates(self)->AbstractSet[str]:
        '''Return the (non-modifiable) set of initial states.'''
        return frozenset(self._initialStates)

    def makeFinalState(self, s: str, acceptanceSets: Optional[Set[str]] = None):
        '''Make state s a final state. If the optional acceptanceSets is provided, s is added to the given acceptance sets. s is assumed to be a state of the automaton.'''
        if not s in self._states:
            raise Exception("{} is not a state of the automaton".format(s))
        if acceptanceSets is None:
            self._finalStates.add(s)
        else:
            for a in acceptanceSets:
                if a=='default':
                    self._finalStates.add(s)
                else:
                    if not a in self._generalizedAcceptanceSets:
                        self._generalizedAcceptanceSets[a] = set()
                    self._generalizedAcceptanceSets[a].add(s)

    def clearInitialStates(self):
        '''Make all states non-initial.'''
        self._initialStates = set()

    def clearFinalStates(self):
        '''Make all states non-final, remove all generalized acceptance sets.'''
        self._finalStates = set()
        self._generalizedAcceptanceSets = dict()

    def finalStates(self)->AbstractSet[str]:
        '''Return the (non-modifiable) set of final states.'''
        return frozenset(self._finalStates)

    def makeNonFinalState(self, s: str):
        '''Make s a non-final state. Assumes s is a state of the automaton.'''
        if not s in self._states:
            raise Exception("{} is not a state of the automaton".format(s))
        if s in self._finalStates:
            self._finalStates.remove(s)

    def acceptsWithPath(self, word: str)->Tuple[bool,Optional[List[str]]]:
        """
            Check if the automaton accepts the given word (a single string of symbols separated by commas).
            Returns a tuple with:
            - boolean indicating if the word is accepted
            - if the previous result is True, the second element is a list of states with an accepting path for the give word. If the word is not accepted, None is returned.
        """

        # get the individual symbols from the input string
        symbols = [] if word=='' else word.split(',')
        
        # loop invariant: maintain set of states reachable by the symbols processed so far and corresponding paths
        # initialize to the epsilon closure of the initial states.
        currentStates: Set[str]
        currentPaths: Dict[str,List[str]]
        currentStates, currentPaths = self.epsilonClosureWithPaths(self._initialStates)

        for symbol in symbols:
            currentStates, paths = self.setNextStatesEpsilonClosureWithPaths(currentStates, symbol)
            newPaths: Dict[str,List[str]] = dict()
            for s in currentStates:
                newPaths[s] = (currentPaths[paths[s][0]])[:-1] + paths[s]
            currentPaths = newPaths

        reachableFinalStates = currentStates.intersection(self._finalStates)
        res = len(reachableFinalStates) != 0
        if res:
            # take arbitrary reachable state
            s = next(iter(reachableFinalStates))
            return res, currentPaths[s]
        else:
            return res, None

    def accepts(self, word: str)->bool:
        """
            Check if the automaton accepts the given word (a single string of symbols separated by commas).            
        """
        res, _ = self.acceptsWithPath(word)
        return res
    
    def isDeterministic(self)->bool:
        '''Check if the automaton is deterministic.'''
        if len(self._initialStates) > 1:
            return False
        if len(self._epsilonTransitions.keys()) > 0:
            return False
        for symbols in self._transitions.values():
            for nextStates in symbols.values():
                if len(nextStates) > 1:
                    return False
        return True

    def asDFA(self)->'Automaton':
        '''Convert to a deterministic automaton.'''
        result = Automaton()

        # convert set of states to string
        def setAsState(ss: AbstractSet[str]): return "{" + (",".join(sorted(ss))) + "}"

        # determine the set of reachable states
        statesExplored: Set[AbstractSet[str]] = set()
        statesToExplore: Set[AbstractSet[str]] = set()
        statesToExplore.add(
            frozenset(self.epsilonClosure(self._initialStates)))

        while statesToExplore != set():
            state = statesToExplore.pop()
            statesExplored.add(state)
            result.addState(setAsState(state))
            symbols = reduce(lambda _symbol, _state: _symbol.union(
                self.outgoingSymbols(_state)), state, set())
            for s in symbols:
                _nextState = frozenset(self.epsilonClosure(
                    self._setNextStates(state, s)))
                if not _nextState in statesExplored:
                    statesToExplore.add(_nextState)
                result.addTransition(setAsState(state), s,
                                     setAsState(_nextState))

        # determine the initial state
        result.makeInitialState(setAsState(
            frozenset(self.epsilonClosure(self._initialStates))))

        # determine the final states
        for s in statesExplored:
            if any(t in self._finalStates for t in s):
                result.makeFinalState(setAsState(s))

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

        sinkStateAdded = False
        sinkState = None

        alphabet = self.alphabet()
        for s in sorted(self._states):
            result.addState(s)
            if s in self._initialStates:
                result.makeInitialState(s)
            if s in self._finalStates:
                result.makeFinalState(s)

            for symbol in sorted(alphabet):
                if self.hasProperTransitionFromStateWithSymbol(s, symbol):
                    for t in sorted(self._transitions[s][symbol]):
                        result.addTransition(s, symbol, t)
                else:
                    if not sinkStateAdded:
                        sinkStateAdded = True
                        sinkState = result.addStateUnique("S")
                    result.addTransition(s, symbol, sinkState)  # type: ignore

        # if a new state was added it needs outgoing transitions to itself
        if sinkStateAdded:
            for symbol in sorted(alphabet):
                result.addTransition(sinkState, symbol, sinkState)  # type: ignore

        return result

    def complement(self)->'Automaton':
        '''Returns the complement of the automaton.'''
        # obtain a deterministic, complete automaton first
        result = self.asDFA().complete()
        # invert the accepting set
        for s in result._states:
            if s in result._finalStates:
                result.makeNonFinalState(s)
            else:
                result.makeFinalState(s)
        return result

    def product(self, A: 'Automaton')->'Automaton':
        '''Return the product automaton with automaton A. The automata synchronize on transitions with symbols common to their alphabets. The automata can independently make transitions on symbols that do not occur in the alphabet of the other automaton.'''
        result = Automaton()

        # figure out the alphabet situation
        myAlphabet = self.alphabet()
        herAlphabet = A.alphabet()
        sharedAlphabet = myAlphabet.intersection(herAlphabet)
        myPrivateAlphabet = myAlphabet.difference(sharedAlphabet)
        herPrivateAlphabet = herAlphabet.difference(sharedAlphabet)

        def prodState(s, t): return "({},{})".format(s, t)

        # create the cartesian product states
        for s in self._states:
            for t in A._states:
                newState = prodState(s, t)
                result.addState(newState)
                if s in self._initialStates and t in A._initialStates:
                    result.makeInitialState(newState)
                if s in self._finalStates and t in A._finalStates:
                    result.makeFinalState(newState)

        # create the transitions
        for s in self._states:
            for t in A._states:
                # my private alphabet transitions
                for symbol in myPrivateAlphabet:
                    for sPrime in self.nextStates(s, symbol):
                        result.addTransition(
                            prodState(s, t), symbol, prodState(sPrime, t))
                # my epsilon transitions
                for sPrime in self.nextEpsilonStates(s):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(sPrime, t))
                # her private alphabet transitions
                for symbol in herPrivateAlphabet:
                    for tPrime in A.nextStates(t, symbol):
                        result.addTransition(
                            prodState(s, t), symbol, prodState(s, tPrime))
                # her epsilon transitions
                for tPrime in A.nextEpsilonStates(t):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(s, tPrime))
                # our common transitions
                for symbol in sharedAlphabet:
                    for sPrime in self.nextStates(s, symbol):
                        for tPrime in A.nextStates(t, symbol):
                            result.addTransition(
                                prodState(s, t), symbol, prodState(sPrime, tPrime))
        return result

    def strictProduct(self, A: 'Automaton')->'Automaton':
        '''Return the 'strict' product automaton with automaton A. The automata synchronize on transitions with symbols common to their alphabets. The automata cannot make transitions on symbols that do not occur in the alphabet of the other automaton.'''

        result = Automaton()

        # figure out the alphabet situation
        myAlphabet = self.alphabet()
        herAlphabet = A.alphabet()
        sharedAlphabet = myAlphabet.intersection(herAlphabet)

        def prodState(s, t): return "({},{})".format(s, t)

        # create the cartesian product states
        for s in self._states:
            for t in A._states:
                newState = prodState(s, t)
                result.addState(newState)
                if s in self._initialStates and t in A._initialStates:
                    result.makeInitialState(newState)
                if s in self._finalStates and t in A._finalStates:
                    result.makeFinalState(newState)

        # create the transitions
        for s in self._states:
            for t in A._states:
                # my epsilon transitions
                for sPrime in self.nextEpsilonStates(s):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(sPrime, t))
                # her epsilon transitions
                for tPrime in A.nextEpsilonStates(t):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(s, tPrime))
                # our common transitions
                for symbol in sharedAlphabet:
                    for sPrime in self.nextStates(s, symbol):
                        for tPrime in A.nextStates(t, symbol):
                            result.addTransition(
                                prodState(s, t), symbol, prodState(sPrime, tPrime))
        return result

    def productBuchi(self, A: 'Automaton')->'Automaton':
        '''Return the product Büchi automaton with Büchi automaton A. The automata synchronize on transitions with symbols common to their alphabets. The automata can independently make transitions on symbols that do not occur in the alphabet of the other automaton.'''
        result = Automaton()

        # figure out the alphabet situation
        myAlphabet = self.alphabet()
        herAlphabet = A.alphabet()
        sharedAlphabet = myAlphabet.intersection(herAlphabet)
        myPrivateAlphabet = myAlphabet.difference(sharedAlphabet)
        herPrivateAlphabet = herAlphabet.difference(sharedAlphabet)

        def prodState(s, t): return "({},{})".format(s, t)

        # create the cartesian product states
        herAcceptanceSet = set()
        for s in self._states:
            for t in A._states:
                newState = prodState(s, t)
                result.addState(newState)
                if s in self._initialStates and t in A._initialStates:
                    result.makeInitialState(newState)
                # take the acceptance conditions from self
                # record the acceptance conditions from her to be added later
                if s in self._finalStates:
                    result.makeFinalState(newState)
                if t in A._finalStates:
                    herAcceptanceSet.add(newState)

        # create the transitions
        for s in self._states:
            for t in A._states:
                # my private alphabet transitions
                for symbol in myPrivateAlphabet:
                    for sPrime in self.nextStates(s, symbol):
                        result.addTransition(
                            prodState(s, t), symbol, prodState(sPrime, t))
                # my epsilon transitions
                for sPrime in self.nextEpsilonStates(s):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(sPrime, t))
                # her private alphabet transitions
                for symbol in herPrivateAlphabet:
                    for tPrime in A.nextStates(t, symbol):
                        result.addTransition(
                            prodState(s, t), symbol, prodState(s, tPrime))
                # her epsilon transitions
                for tPrime in A.nextEpsilonStates(t):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(s, tPrime))
                # our common transitions
                for symbol in sharedAlphabet:
                    for sPrime in self.nextStates(s, symbol):
                        for tPrime in A.nextStates(t, symbol):
                            result.addTransition(
                                prodState(s, t), symbol, prodState(sPrime, tPrime))
        return result.addGeneralizedBuchiAcceptanceSets(set([frozenset(herAcceptanceSet)]))

    def strictProductBuchi(self, A: 'Automaton')->'Automaton':
        '''Return the 'strict' product Büchi automaton with büchi automaton A. The automata synchronize on transitions with symbols common to their alphabets. The automata cannot make transitions on symbols that do not occur in the alphabet of the other automaton.'''
        result = Automaton()

        # figure out the alphabet situation
        myAlphabet = self.alphabet()
        herAlphabet = A.alphabet()
        sharedAlphabet = myAlphabet.intersection(herAlphabet)

        def prodState(s, t): return "({},{})".format(s, t)

        # create the cartesian product states
        # herAcceptanceSet = set()
        for s in self._states:
            for t in A._states:
                newState = prodState(s, t)
                result.addState(newState)
                if s in self._initialStates and t in A._initialStates:
                    result.makeInitialState(newState)
                # determine the generalized acceptance sets
                acceptanceSets = set()
                if s in self._finalStates:
                    acceptanceSets.add("A")
                if t in A._finalStates:
                    acceptanceSets.add("B")
                for accSet in self._generalizedAcceptanceSets.keys():
                    if s in self._generalizedAcceptanceSets[accSet]:
                        acceptanceSets.add("A_" + accSet)
                for accSet in A._generalizedAcceptanceSets.keys():
                    if s in A._generalizedAcceptanceSets[accSet]:
                        acceptanceSets.add("B_" + accSet)
                result.makeFinalState(newState, acceptanceSets)

        # create the transitions
        for s in self._states:
            for t in A._states:
                # my epsilon transitions
                for sPrime in self.nextEpsilonStates(s):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(sPrime, t))
                # her epsilon transitions
                for tPrime in A.nextEpsilonStates(t):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(s, tPrime))
                # our common transitions
                for symbol in sharedAlphabet:
                    for sPrime in self.nextStates(s, symbol):
                        for tPrime in A.nextStates(t, symbol):
                            result.addTransition(
                                prodState(s, t), symbol, prodState(sPrime, tPrime))
        
        return result

    def languageEmpty(self)->Tuple[bool,Optional[List[str]],Optional[List[str]]]:
        ''' Checks if the FSA language is empty. Returns in addition to a Boolean, an accepted word and path if the language is not empty.'''

        # explore if a final state is reachable from an initial state

        # check if one of the initial states is final
        for s in sorted(self._initialStates):
            if s in self._finalStates:
                return (False, [], [s])

        # non-final states that remain to be explored
        statesToExplore = sorted(list(self._initialStates))
        # invariant: states that have already been explored, should all be keys in backTrack
        statesExplored: Set[str] = set()
        # keep track of incoming symbol and state
        backTrack: Dict[str,Tuple[str,str]] = dict()
        while len(statesToExplore) > 0:
            state = statesToExplore.pop(0)
            statesExplored.add(state)
            # for all epsilon transitions
            for s in sorted(self.nextEpsilonStates(state)):
                if not s in statesExplored:
                    statesToExplore.append(s)
                    backTrack[s] = (self._epsilonSymbol, state)
                    if s in self._finalStates:
                        word, path = self._traceAcceptingWordAndPath(s, backTrack)
                        return False, word, path

            # for all symbol transitions
            for symbol in sorted(self.outgoingSymbols(state)):
                for s in sorted(self.nextStates(state, symbol)):
                    if not s in statesExplored:
                        statesToExplore.append(s)
                        backTrack[s] = (symbol, state)
                    if s in self._finalStates:
                        word, path = self._traceAcceptingWordAndPath(s, backTrack)
                        return False, word, path
        # no final state was reached
        return (True, None, None)

    def languageEmptyBuchi(self)->Tuple[bool,Optional[List[str]],Optional[List[str]],Optional[List[str]],Optional[List[str]]]:
        ''' Checks if the Buchi language is empty. Returns an accepted word 9prefix, repetition) and path (prefix, repetition) if the language is not empty.'''

        # Step 1: find all reachable accepting states and determine a word sigma that leads to it

        # Step 2: For each accepting state s do:
        #   - find all states that can be reached with one non-epsilon symbol and remember that symbol a (needed to ensure the final result is an infinite word)
        #   - check if s can be reached from any of those states by a word tau
        #   - (TODO: it may be OK to stop searching when we reaching any of the accepting states we already covered)
        #   - if yes, the automaton accepts the word (sigma)(a tau)**

        # Step 1, determine the reachable accepting states
        reachableStates: Set[str]
        words: Dict[str,List[str]]
        paths: Dict[str,List[str]]
        reachableStates, words, paths = self.reachableStatesWithWordsAndPaths()
        reachableFinalStates: Set[str] = reachableStates.intersection(self._finalStates)

        # Step 2, check for each reachable accepting state if there is a cycle with a non-empty word that returns to it
        for s in sorted(reachableFinalStates):
            # to ensure that the cycle word is non-empty, first determine the states reachable from s with a non-epsilon symbol, and remember that symbol for each state 
            sClosure = self.epsilonClosure(set([s]))
            finalPlusSymbolReachableStates: Set[str] = set()
            singleSymbol: Dict[str,List[str]] = dict()
            for t in sorted(sClosure):
                for symbol in sorted(self.outgoingSymbols(t)):
                    states = self.epsilonClosure(self.nextStates(t, symbol))
                    finalPlusSymbolReachableStates.update(states)
                    for u in sorted(states):
                        singleSymbol[u] = [symbol]

            # test if s is reachable from any state in finalPlusSymbolReachableStates
            cycleReachableStates, cycleWords, cyclePaths = self.reachableStatesWithWordsAndPaths(finalPlusSymbolReachableStates, singleSymbol)
            if s in cycleReachableStates:
                return (False, words[s], cycleWords[s], paths[s], cyclePaths[s])

        # No Cycle found
        return (True, None, None, None, None)

    def languageIncluded(self, A: 'Automaton')->Tuple[bool,Optional[List[str]]]:
        '''Check if the language of the automaton is included in the language of automaton A.
        If not, a word is returned that is in the language of the automaton, but not in the language of A'''
        AC = A.complement()
        P = AC.strictProduct(self)
        boolResult, word, _ = P.languageEmpty()
        return (boolResult, word)

    def subAutomaton(self, states: Set[str])->'Automaton':
        ''' return a sub-automaton containing only the states in the set states '''

        result = Automaton()
        # make states
        for s in self._states.intersection(states):
            result.addState(s)
            if s in self._initialStates:
                result.makeInitialState(s)
            if s in self._finalStates:
                result.makeFinalState(s)

        # make epsilon transitions
        for s in self._epsilonTransitions.keys():
            if s in states:
                for t in self._epsilonTransitions[s]:
                    if t in states:
                        result.addEpsilonTransition(s, t)

        # make regular transitions
        for s in self._transitions.keys():
            if s in states:
                for symbol in self._transitions[s]:
                    for t in self._transitions[s][symbol]:
                        if t in states:
                            result.addTransition(s, symbol, t)

        return result

    def eliminateReachability(self)-> 'Automaton':
        '''Reduce the size of the automaton by removing unreachable states and states from which no final state is reachable.'''

        # remove unreachable states
        states = self.reachableStates().intersection(self.reachableStatesFinal())
        return self.subAutomaton(states)

    def eliminateStatesWithoutOutgoingTransitions(self)-> 'Automaton':
        '''Return an automaton where all states without outgoing transitions are removed. For Büchi automata this results in an equivalent automaton.'''
        toEliminate: Set[str] = set()
        for s in self._states:
            if len(self.outgoingSymbolsWithEpsilon(s)) ==0:
                toEliminate.add(s)

        if len(toEliminate) ==0:
            return self

        return self.subAutomaton(self._states.difference(toEliminate)).eliminateStatesWithoutOutgoingTransitions()

    def _partitionRefinement(self)->Tuple[Set[AbstractSet[str]],Dict[str,AbstractSet[str]]]:
        '''Return equivalence classes according to a partition refinement process.'''

        def _createPartition(partitions: Set[AbstractSet[str]], partitionMap: Dict[str,AbstractSet[str]], setOfStates: Set[str]):
            '''Create a partition for setOfStates, add it to partitions and update the partition map accordingly.'''
            fSetOfStates = frozenset(setOfStates)
            partitions.add(fSetOfStates)
            for s in setOfStates:
                partitionMap[s] = fSetOfStates

        def _partitionRefinementEdgesEquivalent(s1: str, s2: str)->bool:
            '''Check if states s1 and s2 are considered equivalent.'''

            # s1 and s2 are equivalent if for every s1-a->C, s2-a->C and vice versa
            labels: Set[str] = set()
            labels.update(self.outgoingSymbolsSet(self.epsilonClosure(set([s1]))))
            labels.update(self.outgoingSymbolsSet(self.epsilonClosure(set([s2]))))

            ecs1 = self.epsilonClosure(set([s1]))
            ecs2 = self.epsilonClosure(set([s2]))

            # for every label, compare outgoing edges
            for l in labels:
                # collect classes of states in ns1 and ns2
                cs1: Set[AbstractSet[str]] = set()
                cs2: Set[AbstractSet[str]] = set()
                for t in ecs1:
                    for s in self.epsilonClosure(self.nextStates(t, l)):
                        cs1.add(partitionMap[s])
                for t in ecs2:
                    for s in self.epsilonClosure(self.nextStates(t, l)):
                        cs2.add(partitionMap[s])
                # compare classes
                if cs1 != cs2:
                    return False

            return True

        # make initial partition on states that agree on final-ness
        partitions: Set[AbstractSet[str]] = set()
        # final are states from which a final state is reachable with epsilon moves
        states_f = self._backwardEpsilonClosure(self._finalStates)
        # non-final, others
        states_nf = self._states.difference(states_f)
        p_f = None
        if len(states_f):
            p_f = frozenset(states_f)
            partitions.add(p_f) 
        p_nf = None
        if len(states_nf):
            p_nf = frozenset(states_nf)
            partitions.add(p_nf)

        partitionMap: Dict[str,AbstractSet[str]] = dict()
        for s in states_f:
            partitionMap[s] = p_f  # type: ignore I'm sure p_f is not None here
        for s in states_nf:
            partitionMap[s] = p_nf  # type: ignore I'm sure p_nf is not None here

        oldPartitions: Set[AbstractSet[str]] = set()

        while len(oldPartitions) != len(partitions):
            # print(partitions)
            newPartitions: Set[AbstractSet[str]] = set()
            for eClass in partitions:
                # pick arbitrary state from class
                s1 = next(iter(eClass))

                equivSet: Set[str] = set()
                remainingSet: Set[str] = set()
                equivSet.add(s1)

                # check whether all other states can go with the same labels to
                # the same set of other equivalence classes.
                for s2 in eClass:
                    if s2 != s1:
                        if _partitionRefinementEdgesEquivalent(s1, s2):
                            equivSet.add(s2)
                        else:
                            remainingSet.add(s2)
                
                # if not, split the class
                if len(equivSet) == len(eClass):
                    _createPartition(newPartitions, partitionMap, equivSet)
                else:
                    _createPartition(newPartitions, partitionMap, equivSet)
                    _createPartition(newPartitions, partitionMap, remainingSet)

            oldPartitions = partitions
            partitions = newPartitions

        return partitions, partitionMap

    def minimize(self)->'Automaton':
        '''Implements a partition refinement strategy to reduce the size of the FSA.'''
        
        def setAsState(ss): return "{" + (",".join(sorted(ss))) + "}"

        # remove unreachable states
        # remove states from which final states are not reachable
        interim = self.eliminateReachability()
        # find equivalent states through partition refinement.
        partitions, partitionMap = interim._partitionRefinement()

        result = Automaton()
 
        # make states
        for p in partitions:
            ns = setAsState(p)
            result.addState(ns)
            s = next(iter(p))
            if s in interim._backwardEpsilonClosure(interim._finalStates):
                result.makeFinalState(ns)

            # determine initial states
            # a partition is initial if one of its states was initial
            for s in p:
                if s in interim._initialStates:
                    result.makeInitialState(ns)

        # make transitions
        for p in partitions:
            # take a representative state
            s = next(iter(p))
            for t in interim.epsilonClosure(set([s])):
                for symbol in interim.outgoingSymbols(t):
                    for u in interim.nextStates(t, symbol):
                        result.addTransition(setAsState(partitionMap[s]), symbol, setAsState(partitionMap[u]))
                if t in interim._epsilonTransitions:
                    for u in interim._epsilonTransitions[t]:
                        if partitionMap[s] != partitionMap[u]:
                            result.addEpsilonTransition(setAsState(partitionMap[s]), setAsState(partitionMap[u]))
        return result

    def minimizeBuchi(self)->'Automaton':
        '''Implements a partition refinement strategy to reduce the size of the Büchi automaton.'''
        
        # eliminate states from which not all acceptance sets are reachable
        interim = self.eliminateStatesWithoutOutgoingTransitions()
        return  interim.minimize()

    def statesInBFSOrder(self)->List[str]:
        '''Return a list of state in breadth-first order'''
        result = list()
        self._breadthFirstSearch(lambda s: result.append(s))
        return result
        
    def relabelStates(self)->'Automaton':
        '''Return the automaton with states relabeled 'S' with a number in a breadth first manner.'''

        def _stateName(n:int)->str:
            return "S{}".format(n)

        def _createState(s: str):
            nonlocal n
            newState = _stateName(n)
            stateDict[s] = newState
            result.addState(newState)
            n += 1

        result = Automaton()
        stateDict = dict()
        n = 1

        self._breadthFirstSearch(_createState)

        for i in self._initialStates:
            result.makeInitialState(stateDict[i])

        for f in self._finalStates:
            result.makeFinalState(stateDict[f])

        for s in self._epsilonTransitions.keys():
            for t in self._epsilonTransitions[s]:
                result.addEpsilonTransition(stateDict[s], stateDict[t])

        for s in self._transitions.keys():
            for symbol in self._transitions[s].keys():
                for t in self._transitions[s][symbol]:
                    result.addTransition(stateDict[s], symbol, stateDict[t])
        return result

    def eliminateEpsilonTransitions(self)->'Automaton':
        '''Eliminate epsilon transitions from the automaton.'''
        
        result = Automaton()
        
        for s in self._states:
            result.addState(s)
        
        for s in self._transitions.keys():
            for symbol in self._transitions[s].keys():
                for t in self._transitions[s][symbol]:
                    # for u in self.epsilonClosure(set([t])):
                    for v in self._backwardEpsilonClosure(set([s])):
                        result.addTransition(v, symbol, t)

        for s in self._states:
            if s in self._initialStates:
                result.makeInitialState(s)
                # for t in self.epsilonClosure(set([s])):
                #     result.makeInitialState(t)
            if s in self._finalStates:
                for t in self._backwardEpsilonClosure(set([s])):
                    result.makeFinalState(t)
        return result


    def _asDSLSymbol(self, symbol: str)->str:
        '''Escape quotes.'''
        if re.match(r"[a-zA-Z][a-zA-Z0-9]*", symbol):
            return symbol
        return '"' + symbol.replace('"', '\\"') + '"'

    def asDSL(self, name: str)->str:
        '''Return a string representing the automaton in the domain-specific language,'''

        def _addStateWithAttributes(u: str, statesOutput:Set[str], output: StringIO):
            output.write(u)
            if not u in statesOutput:
                # if it is the first time we are using this state, add its attributes
                self._dslOutputStateAttributes(u, output)
                statesOutput.add(u)

        # keep track of the states that have been output
        statesOutput: Set[str] = set()
        # create string writer for the output
        output = StringIO()
        # write header
        output.write("finite state automaton {} {{\n".format(name))
        # for all transitions (collecting multiple transitions into one)
        for (s, symbols, t) in sorted(self._dslMultiSymbolTransitions()):
            # write the transition
            output.write("\t")
            _addStateWithAttributes(s, statesOutput, output)
            if len(symbols)==1 and self._epsilonSymbol in symbols:
                output.write(" ----> ")
            else:
                output.write(" -- ")
                output.write(", ".join([self._asDSLSymbol(symbol) for symbol in sorted(symbols)]))
                output.write(" --> ")
            _addStateWithAttributes(t, statesOutput, output)
            output.write("\n")
        # write the remaining states without transitions
        remainingStates = self._states.difference(statesOutput)
        if len(remainingStates):
            output.write("\tstates\n")
            for s in remainingStates:
                output.write("\t")
                _addStateWithAttributes(s, statesOutput, output)
                output.write("\n")
        output.write("}\n")
        result = output.getvalue()
        output.close()
        return result

    @staticmethod
    def _ParsingAddStateWithLabels(fsa: 'Automaton', s: str, labels:Set[str], acceptanceSets:Set[str]):
        '''Add a state with the given attributes.'''
        fsa.addState(s)
        for attr in labels:
            if attr == 'initial' or attr == 'i':
                fsa.makeInitialState(s)
            if attr == 'final' or attr == 'f':
                fsa.makeFinalState(s, acceptanceSets)

    @staticmethod
    def fromDSL(dslString)->Tuple[str,'Automaton']:
        '''Create an automaton from the DSL string.'''

        factory = dict()
        factory['Init'] = lambda : Automaton()
        factory['addTransitionPossiblyEpsilon'] = lambda fsa, s, t, symbol: \
            fsa.addEpsilonTransition(s, t) if symbol == Automaton._epsilonSymbol else fsa.addTransition(s, symbol, t)
        factory['AddEpsilonTransition'] = lambda fsa, s, t : fsa.addEpsilonTransition(s, t)
        factory['AddState'] = lambda fsa, s, labels, acceptanceSets: Automaton._ParsingAddStateWithLabels(fsa, s, labels, acceptanceSets)
        name, fsa = parseFSADSL(dslString, factory)
        if name is None or fsa is None:
            exit(1)
        return name, fsa
        
    def reachableStates(self)->Set[str]:
        ''' return a set of all states reachable from an initial state '''
        result, _, _ = self.reachableStatesWithWordsAndPaths()
        return result

    def reachableStatesWithWordsAndPaths(self, startingStates:Optional[Iterable[str]]=None, startingWords:Optional[Dict[str,List[str]]]=None, startingPaths=None)->Tuple[Set[str],Dict[str,List[str]],Dict[str,List[str]]]:
        ''' return a set of all states reachable from any state in startingStates and for each a word and a path by which is is reached. If startingStates is omitted, the initial states are used'''
        if startingStates is None:
            startingStates = sorted(self._initialStates)
        result = set()
        words: Dict[str,List[str]]
        if startingWords is None:
            words = dict()
            for  s in startingStates:
                words[s] = []
        else:
            words = startingWords
        if startingPaths is None:
            paths = dict()
            for  s in startingStates:
                paths[s] = [s]
        else:
            paths = startingPaths
        statesToExplore = sorted(list(startingStates))
        while len(statesToExplore) > 0:
            s = statesToExplore.pop()
            result.add(s)
            if s in self._epsilonTransitions:
                for t in sorted(self._epsilonTransitions[s]):
                    if not t in result:
                        statesToExplore.append(t)
                        words[t] = words[s]
                        paths[t] = paths[s] +[t]
            if s in self._transitions:
                for symbol in sorted(self._transitions[s]):
                    for t in sorted(self._transitions[s][symbol]):
                        if not t in result:
                            statesToExplore.append(t)
                            words[t] = words[s] + [symbol]
                            paths[t] = paths[s] + [t]

        return result, words, paths

    def reachableStatesFinal(self)->Set[str]:
        ''' return the set of all states from which a final state is reachable '''
        result: Set[str] = set()
        statesToExplore = set(self._finalStates)
        while statesToExplore != set():
            s = statesToExplore.pop()
            result.add(s)
            # check epsilon transitions
            for t in self._epsilonTransitions:
                for u in self._epsilonTransitions[t]:
                    if u == s:
                        if not t in result:
                            statesToExplore.add(t)
            # check regular transitions
            for t in self._transitions:
                for symbol in self._transitions[t]:
                    for u in self._transitions[t][symbol]:
                        if u == s:
                            if not t in result:
                                statesToExplore.add(t)
        return result

    def addGeneralizedBuchiAcceptanceSets(self, A:Iterable[AbstractSet[str]])->'Automaton':
        '''
        return a new non-generalized Büchi automaton with the added generalized acceptance sets incorporated
        '''
        res, _ = self.addGeneralizedBuchiAcceptanceSetsWithStateMap(A)
        return res

    def addGeneralizedBuchiAcceptanceSetsWithStateMap(self, A:Iterable[AbstractSet[str]])->Tuple['Automaton',Dict[str,str]]:
        '''
        return a new non-generalized Büchi automaton with the added generalized acceptance sets incorporated and a map linking the new states to the original states
        '''

        def _newState(s: str, n: int)->str:
            return "({},F{})".format(s,str(n))
       
        stateMap:Dict[str,str] = dict()

        # create a copy of every state for every acceptance set.
        # label final state accordingly
        # add transitions to state in same layer for non-accepting source states 
        # or state in next layer if it is accepting
        res = Automaton()
        acceptanceSets:List[AbstractSet[str]] = []
        if len(self._finalStates) > 0:
            acceptanceSets.append(self._finalStates)
        for a in A:
            acceptanceSets.append(a)
        N = len(acceptanceSets)

        # create states
        for n in range(N):
            for s in self._states:
                ns = _newState(s,n)
                stateMap[ns] = s
                res.addState(ns)

        # set initial states
        for s in self._initialStates:
            res.makeInitialState(_newState(s,0))

        # set final state s
        for n in range(N):
            for s in acceptanceSets[n]:
                res.makeFinalState(_newState(s,n))

        # add transitions
        for n in range(N):
            nxt = (n+1)%N
            for s in self._transitions:
                for symbol in self._transitions[s]:
                    for t in self._transitions[s][symbol]:
                        if (s in acceptanceSets[n]):
                            res.addTransition(_newState(s,n), symbol, _newState(t,nxt))
                        else:
                            res.addTransition(_newState(s,n), symbol, _newState(t,n))
            for s in self._epsilonTransitions:
                for t in self._epsilonTransitions[s]:
                    if (s in acceptanceSets[n]):
                        res.addEpsilonTransition(_newState(s,n), _newState(t,nxt))
                    else:
                        res.addEpsilonTransition(_newState(s,n), _newState(t,n))

        return res, stateMap

    def hasGeneralizedAcceptanceSets(self)->bool:
        '''Test if the automaton has generalized acceptance sets.'''
        return len(self._generalizedAcceptanceSets) > 0


    def asRegularBuchiAutomaton(self)->'Automaton':
        '''Convert to an equivalent regular Büchi automaton, i.e., without generalized acceptance sets.'''
        res, _ = self.asRegularBuchiAutomatonWithStateMap()
        return res

    def asRegularBuchiAutomatonWithStateMap(self)->Tuple['Automaton',Dict[str,str]]:
        '''Convert to an equivalent regular Büchi automaton, i.e., without generalized acceptance sets. Return automaton and a map relating the states in the new automaton to states in the old automaton.'''
        if len(self._generalizedAcceptanceSets) == 0:
            stateMap = dict()
            for s in self._states:
                stateMap[s] = s
            return self, stateMap
        return self.addGeneralizedBuchiAcceptanceSetsWithStateMap(list(self._generalizedAcceptanceSets.values()))

    def _dslMultiSymbolTransitions(self)->Set[Tuple[str,Tuple[str],str]]:
        '''collect common transitions into multi-labels, including epsilon transitions'''
        reorg = dict()

		# handle all regular transitions
        for s in self._transitions.keys():
			# ensure s is a key in reorg
            if not s in reorg.keys():
                reorg[s] = dict()

			# for all transitions from s collect the symbols going to states t
            for symbol in self._transitions[s]:
                for t in self._transitions[s][symbol]:
					# ensure that t is a key in reorg[s]
                    if not t in reorg[s].keys():
                        reorg[s][t] = set()
					# add the symbol to the set
                    reorg[s][t].add(symbol)

		# handle all epsilon transitions
        for s in self._epsilonTransitions.keys():
			# ensure s is a key in reorg
            if not s in reorg.keys():
                reorg[s] = dict()
            for t in self._epsilonTransitions[s]:
				# ensure that t is a key in reorg[s]
                if not t in reorg[s].keys():
                    reorg[s][t] = set()
				# add the symbol to the set
                reorg[s][t].add(self._epsilonSymbol)

		# create the results
        result: Set[Tuple[str,Tuple[str],str]] = set()
        for s in reorg.keys():
            for t in reorg[s].keys():
                result.add((s, tuple(sorted(reorg[s][t])), t))
        return result

    def _dslOutputStateAttributes(self, state: str, output: StringIO):
        '''Output state attributes to output'''
        if state in self._initialStates:
            output.write(" initial")
            if state in self._finalStates:
                output.write("; final")
            return
        generalizedSets = {a for a in self._generalizedAcceptanceSets if state in self._generalizedAcceptanceSets[a]}
        if state in self._finalStates or len(generalizedSets)>0:
            output.write(" final")
        if len(generalizedSets)>0:
            if state in self._finalStates:
                generalizedSets.add('default')
            output.write(" [{}]".format(", ".join(generalizedSets)))

    def outgoingSymbols(self, state: str)->AbstractSet[str]:
        '''Return the set of outgoing symbols from state.'''
        if not state in self._transitions:
            return set()
        return frozenset(self._transitions[state].keys())

    
    def outgoingSymbolsSet(self, setOfStates: Set[str])->AbstractSet[str]:
        '''Return the set of outgoing symbols from any state from setOfStates.'''
        res = set()
        for s in setOfStates:
            res.update(self.outgoingSymbols(s))
        return frozenset(res)

    def outgoingSymbolsWithEpsilon(self, state: str)->AbstractSet[str]:
        '''Return the set of outgoing symbols from state, including # for epsilon transitions.'''
        result = set(self.outgoingSymbols(state))
        if state in self._epsilonTransitions:
            result.add(Automaton._epsilonSymbol)
        return frozenset(result)

    def nextStates(self, state: str, symbol: str)->Set[str]:
        """
            Return the set of states reachable from 'state' by a transition labelled 'symbol', where symbol is a non-epsilon symbol.
        """
        if not state in self._transitions:
            return set()
        if not symbol in self._transitions[state]:
            return set()
        return self._transitions[state][symbol]

    def nextStatesWithEpsilon(self, state: str, symbol: str)->Set[str]:
        """
            Return the set of states reachable from 'state' by a transition labelled 'symbol', where symbol can be an epsilon symbol.
        """
        if symbol == Automaton._epsilonSymbol:
            return self._epsilonTransitions[state]
        else:
            if not state in self._transitions:
                return set()
            if not symbol in self._transitions[state]:
                return set()
            return self._transitions[state][symbol]

    def nextEpsilonStates(self, state: str)->Set[str]:
        """
            Return the set of states reachable from 'state' by a single epsilon transition.
        """
        if not state in self._epsilonTransitions:
            return set()
        return self._epsilonTransitions[state]

    def nextStatesEpsilonClosureWithPaths(self, state: str, symbol: str)->Tuple[Set[str],Dict[str,List[str]]]:
        """
            Return the set of states reachable from 'state' by a sequence of transitions including an arbitrary number of epsilon transitions and one 'symbol' transition, where symbol is not epsilon.
            Returns a tuple with:
            - a set of states thus reachable
            - a dictionary with for every reachable state s, a path starting in 'state' and ending in s.
        """
        return self.setNextStatesEpsilonClosureWithPaths(set([state]), symbol)

    def setNextStatesEpsilonClosureWithPaths(self, states: Set[str], symbol: str) -> Tuple[Set[str],Dict[str,List[str]]]:
        """
            Return the set of states reachable from a state from 'states' by a sequence of transitions including an arbitrary number of epsilon transitions and one 'symbol' transition, where symbol is not epsilon.
            Returns a tuple with:
            - a set of states thus reachable
            - a dictionary with for every reachable state s, a path starting in a state from 'states' and ending in s.
        """
        preEpsilonReachableStates, prePaths = self.epsilonClosureWithPaths(states)
        afterSymbolStates: Set[str] = set()
        afterSymbolPaths: Dict[str,List[str]] = dict()
        for s in preEpsilonReachableStates:
            for t in self.nextStates(s, symbol):
                afterSymbolStates.add(t)
                afterSymbolPaths[t] = prePaths[s] + [t]
        postEpsilonReachableStates, postPaths = self.epsilonClosureWithPaths(afterSymbolStates)
        resPaths: Dict[str,List[str]] = dict()
        for s in postEpsilonReachableStates:
            resPaths[s] = (afterSymbolPaths[postPaths[s][0]])[:-1] + postPaths[s]
        
        return postEpsilonReachableStates, resPaths


    def _setNextStates(self, states: AbstractSet[str], symbol: str)->Set[str]:
        '''Return the set of states reachable from a state in states with a symbol transition.'''
        nStates = set()
        for s in states:
            nStates.update(self.nextStates(s, symbol))
        return nStates

    def addStateUnique(self, state: str)->str:
        '''Add a state named state, but ensure that it is unique by potentially modifying the name.'''
        if not state in self._states:
            self.addState(state)
            return state
        n = 1
        while state+str(n) in self._states:
            n += 1
        newState = state+str(n)
        self.addState(newState)
        return newState

    def epsilonClosureWithPaths(self, setOfStates: Set[str])->Tuple[Set[str],Dict[str,List[str]]]:
        """
            Determine the epsilon closure of the given set of states. Return a tuple with:
            - the set of states reachable by zero or more epsilon transitions
            - a dictionary that maps each of the reachable states s to a path of states starting from one of the states in the initial set and ending in s.
        """
        res = setOfStates
        paths = dict()
        for s in setOfStates:
            paths[s] = [s]
        n = 0
        while n < len(res):
            n = len(res)
            newRes = res.copy()
            for s in res:
                if s in self._epsilonTransitions:
                    newStates = self._epsilonTransitions[s]
                    newRes.update(newStates)
                    for t in newStates:
                        if t not in paths:
                            paths[t] = paths[s] + [t]
            res = newRes
        return res, paths

    def epsilonClosure(self, setOfStates: Set[str])->Set[str]:
        '''Determine the set of states reachable by epsilon transitions from any state in setOfStates.'''
        res, _ = self.epsilonClosureWithPaths(setOfStates)
        return res

    def _backwardEpsilonClosure(self, setOfStates: Set[str])->Set[str]:
        """
            Determine the backward epsilon closure of the given set of states. Return a set of states reachable by zero or more epsilon transitions taken backward.
        """
        res = setOfStates
        n = 0
        while n < len(res):
            n = len(res)
            newRes = res.copy()
            for s in res:
                for t in self._epsilonTransitions:
                    if s in self._epsilonTransitions[t]:
                        newRes.add(t)
            res = newRes
        return res

    def _traceAcceptingWordAndPath(self, s: str, backTrack: Dict[str,Tuple[str,str]])->Tuple[List[str],List[str]]:
        '''Reconstruct accepting word and path from state s tracking back to an initials tate using backTrack. Return bool for success and if successful word and path.'''
        word = []
        path = [s]
        t = s
        while not t in self._initialStates:
            (sym, t) = backTrack[t]
            if sym != self._epsilonSymbol:
                word.insert(0, sym)
            path.insert(0, t)
        return (word, path)

    def _successorStates(self, s:str)->Set[str]:
        '''Return a set of successor states of s by any transition or epsilon transition.'''
        result = set()
        if s in self._transitions:
            for symbol in self._transitions[s]:
                result.update(self.nextStates(s, symbol))
        if s in self._epsilonTransitions:
            result.update(self._epsilonTransitions[s])
        return result

    def _breadthFirstSearch(self, visit: Callable[[str],None]):
        visitedStates: Set[str] = set()
        statesToVisit = sorted(list(self._initialStates))
        setOfStatesToVisit = set(self._initialStates)
        while len(statesToVisit) > 0:
            s = statesToVisit.pop(0)
            setOfStatesToVisit.remove(s)
            visit(s)
            visitedStates.add(s)
            newStates = self._successorStates(s).difference(visitedStates).difference(setOfStatesToVisit)
            for t in sorted(newStates):
                statesToVisit.append(t)
                setOfStatesToVisit.add(t)

        # remaining states in no particular order
        for s in self._states.difference(visitedStates):
            visit(s)

    def __str__(self)->str:
        return "({}, {}, {}, {})".format(self._states, self._initialStates, self._finalStates, self._transitions)
