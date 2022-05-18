from functools import reduce
from io import StringIO
import re
from finitestateautomata.libfsagrammar import parseFSADSL


class Automaton(object):

    _epsilonSymbol = '#'

    def __init__(self):
        self._states = set()
        self._transitions = dict()
        self._epsilonTransitions = dict()
        self._initialStates = set()
        self._finalStates = set()
        self._generalizedAcceptanceSets = dict()
    
    def addState(self, s):
        self._states.add(s)

    def addStates(self, setOfStates):
        self._states.update(setOfStates)

    def states(self):
        return frozenset(self._states)

    def addTransition(self, srcState, symbol, dstState):
        self.addState(srcState)
        self.addState(dstState)
        if not srcState in self._transitions.keys():
            self._transitions[srcState] = dict()
        if not symbol in self._transitions[srcState].keys():
            self._transitions[srcState][symbol] = set()
        self._transitions[srcState][symbol].add(dstState)

    def hasProperTransitionFromState(self, srcState):
        # check if state srcState has a labelled outgoing transition
        if srcState in self._transitions.keys():
            return len(self._transitions[srcState]) > 0
        else:
            return False

    def hasProperTransitionFromStateWithSymbol(self, srcState, symbol):
        if self.hasProperTransitionFromState(srcState):
            return symbol in self._transitions[srcState].keys()
        else:
            return False

    def transitions(self):
        result = set()
        for src in self._transitions:
            for symbol in self._transitions[src]:
                for dst in self._transitions[src][symbol]:
                    result.add((src, symbol, dst))
        return result

    def groupedTransitions(self):
        result = set()
        trans = self.transitions()
        epsTrans = self.epsilonTransitions()
        statePairs = set([(t[0], t[2]) for t in trans])
        statePairs.update(epsTrans)

        for p in statePairs:
            symbols = [t[1] for t in {u for u in trans if u[0]==p[0] and u[2]==p[1]}]
            if (p[0], p[1]) in epsTrans:
                symbols.append('#') # NameError: "name '_epsilonSymbol' is not defined"
            # sort
            sortedSymbols = sorted(symbols)
            result.add((p[0], ','.join(sortedSymbols), p[1]))
        return result


    def addEpsilonTransition(self, srcState, dstState):
        self.addState(srcState)
        self.addState(dstState)
        if not srcState in self._epsilonTransitions.keys():
            self._epsilonTransitions[srcState] = set()
        self._epsilonTransitions[srcState].add(dstState)

    def epsilonTransitions(self):
        result = set()
        for src in self._epsilonTransitions:
            for dst in self._epsilonTransitions[src]:
                result.add((src, dst))
        return result

    def makeInitialState(self, s):
        if not s in self._states:
            raise Exception("{} is not a state of the automaton".format(s))
        self._initialStates.add(s)

    def initialStates(self):
        return frozenset(self._initialStates)

    def makeFinalState(self, s, acceptanceSets = None):
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
        self._initialStates = set()

    def clearFinalStates(self):
        self._finalStates = set()
        self._generalizedAcceptanceSets = dict()

    def finalStates(self):
        return frozenset(self._finalStates)

    def makeNonFinalState(self, s):
        if not s in self._states:
            raise Exception("{} is not a state of the automaton".format(s))
        if s in self._finalStates:
            self._finalStates.remove(s)

    def acceptsWithPath(self, word):
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
        currentStates, currentPaths = self.epsilonClosureWithPaths(self._initialStates)

        for symbol in symbols:
            currentStates, paths = self.setNextStatesEpsilonClosureWithPaths(currentStates, symbol)
            newPaths = dict()
            for s in currentStates:
                newPaths[s] = (currentPaths[paths[s][0]])[:-1] + paths[s]
            currentPaths = newPaths

        reachableFinalStates = currentStates.intersection(self._finalStates)
        res = len(reachableFinalStates) != 0
        if res:
            # take arbitrary reachable 
            s = next(iter(reachableFinalStates))
            return res, currentPaths[s]
        else:
            return res, None

    def accepts(self, word):
        """
            Check if the automaton accepts the given word.            
        """
        res, _ = self.acceptsWithPath(word)
        return res
    
    def isDeterministic(self):
        if len(self._initialStates) > 1:
            return False
        if len(self._epsilonTransitions.keys()) > 0:
            return False
        for symbols in self._transitions.values():
            for nextStates in symbols.values():
                if len(nextStates) > 1:
                    return False
        return True

    def asDFA(self):
        result = Automaton()

        # convert set of states to string
        def setAsState(ss): return "{" + (",".join(sorted(ss))) + "}"

        # determine the set of reachable states
        statesExplored = set()
        statesToExplore = set()
        statesToExplore.add(
            frozenset(self.epsilonClosure(self._initialStates)))

        while statesToExplore != set():
            state = statesToExplore.pop()
            statesExplored.add(state)
            result.addState(setAsState(state))
            symbols = reduce(lambda _symb, _state: _symb.union(
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

    def alphabet(self):
        '''Return the alphabet of the automaton. I.e., all symbols that occur on transition'''
        result = set()
        for _, trans in self._transitions.items():
            result.update(trans.keys())
        return result

    def complete(self):
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
                    result.addTransition(s, symbol, sinkState)

        # if a new state was added it needs outgoing transitions to itself
        if sinkStateAdded:
            for symbol in sorted(alphabet):
                result.addTransition(sinkState, symbol, sinkState)

        return result

    def complement(self):
        # obtain a deterministic, complete automaton first
        result = self.asDFA().complete()
        # invert the accepting set
        for s in result._states:
            if s in result._finalStates:
                result.makeNonFinalState(s)
            else:
                result.makeFinalState(s)
        return result

    def product(self, A):
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

    def strictProduct(self, A):
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

    def productBuchi(self, A):
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
                # record the acceptance conditiond from her to be added later
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

    def strictProductBuchi(self, B):
        result = Automaton()

        # figure out the alphabet situation
        myAlphabet = self.alphabet()
        herAlphabet = B.alphabet()
        sharedAlphabet = myAlphabet.intersection(herAlphabet)
        myPrivateAlphabet = myAlphabet.difference(sharedAlphabet)
        herPrivateAlphabet = herAlphabet.difference(sharedAlphabet)

        def prodState(s, t): return "({},{})".format(s, t)

        # create the cartesian product states
        # herAcceptanceSet = set()
        for s in self._states:
            for t in B._states:
                newState = prodState(s, t)
                result.addState(newState)
                if s in self._initialStates and t in B._initialStates:
                    result.makeInitialState(newState)
                # determine the generalized acceptance sets
                acceptanceSets = set()
                if s in self._finalStates:
                    acceptanceSets.add("A")
                if t in B._finalStates:
                    acceptanceSets.add("B")
                for accSet in self._generalizedAcceptanceSets.keys():
                    if s in self._generalizedAcceptanceSets[accSet]:
                        acceptanceSets.add("A_" + accSet)
                for accSet in B._generalizedAcceptanceSets.keys():
                    if s in B._generalizedAcceptanceSets[accSet]:
                        acceptanceSets.add("B_" + accSet)
                result.makeFinalState(newState, acceptanceSets)

        # create the transitions
        for s in self._states:
            for t in B._states:
                # my epsilon transitions
                for sPrime in self.nextEpsilonStates(s):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(sPrime, t))
                # her epsilon transitions
                for tPrime in B.nextEpsilonStates(t):
                    result.addEpsilonTransition(
                        prodState(s, t), prodState(s, tPrime))
                # our common transitions
                for symbol in sharedAlphabet:
                    for sPrime in self.nextStates(s, symbol):
                        for tPrime in B.nextStates(t, symbol):
                            result.addTransition(
                                prodState(s, t), symbol, prodState(sPrime, tPrime))
        
        return result

    def strictProductNonGeneralizedBuchi(self, A):
        '''
        Obsolete.
        Compute the strict product of two non-generalized Buchi automata.
        Returns a non-generalized automaton
        '''
        
        if len(self._generalizedAcceptanceSets) > 0 or len(A._generalizedAcceptanceSets) > 0:
            raise Exception("strictProductNonGeneralizedBuchi cannot be used on generalized Büchi automata")

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
        
        return result.addGeneralizedBuchiAcceptanceSets(set([frozenset(herAcceptanceSet)]))

    def languageEmpty(self):
        ''' Checks if the language is empty. Returns an accepted word and path if the language is not empty.'''

        #  say they are ignored  (should only occur for Buchi)
        if len(self._generalizedAcceptanceSets) > 0:
            raise Exception("languageEmpty cannot be used on generalized Büchi automata")

        # explore if a final state is reachable from an initial state

        # check if one of the initial states is final
        for s in sorted(self._initialStates):
            if s in self._finalStates:
                return (False, [], [s])

        # non-final states that remain to be explored
        statesToExplore = sorted(list(self._initialStates))
        # invariant: states that have already been explored, should all be keys in backTrack
        statesExplored = set()
        # keep track of incoming symbol and state
        backTrack = dict()
        while len(statesToExplore) > 0:
            state = statesToExplore.pop(0)
            statesExplored.add(state)
            # for all epsilon transitions
            for s in sorted(self.nextEpsilonStates(state)):
                if not s in statesExplored:
                    statesToExplore.append(s)
                    backTrack[s] = (self._epsilonSymbol, state)
                    if s in self._finalStates:
                        return self._traceAcceptingWordAndPath(s, backTrack)

            # for all symbol transitions
            for symbol in sorted(self.outgoingSymbols(state)):
                for s in sorted(self.nextStates(state, symbol)):
                    if not s in statesExplored:
                        statesToExplore.append(s)
                        backTrack[s] = (symbol, state)
                    if s in self._finalStates:
                        return self._traceAcceptingWordAndPath(s, backTrack)
        # no final state was reached
        return (True, None, None)

    def languageEmptyBuchi(self):
        ''' Checks if the Buchi language is empty. Returns an accepted word if the language is not empty.'''

        def _split(stack1, symbolStack1, s):
            i = stack1.index(s) 
            return symbolStack1[:i+1], symbolStack1[i+1:]

        def _stripEpsilon(word):
            return [l for l in word if l != self._epsilonSymbol]

        # retrun all pairs (t, symb) such that there is a transition from s to t, labelled symb (including the epsilon symbol)
        def _successorStates(s):
            res = set()
            # if s has outgoing transitions
            if s in self._transitions:
                # for all outgoing symbols
                for symb in self._transitions[s]:
                    # for every state t reachable with symbol symb, add (t, symb) to the set
                    res.update({ (t, symb) for t in self._transitions[s][symb]})
            # if s has outgoing epsilon transitions
            if s in self._epsilonTransitions:
                # add those too
                res.update({ (t, self._epsilonSymbol) for t in self._epsilonTransitions[s]})
            return res

        def _notOnlyEpsilon(stack):
            for s in stack:
                if s != self._epsilonSymbol:
                    return True
            return False
        
        def _notOnlyEpsilonStack1(s):
            nonlocal stack1, symbolStack1
            i = stack1.index(s) 
            return _notOnlyEpsilon(symbolStack1[i+1:])


        # ??
        def _exploreDFSCycle(states):
            nonlocal stack2, stack1, symbolStack1, symbolStack2, exploredStates, exploredStatesCycle
            for (s, symb) in states:
                if s in stack1Set:
                    symbolStack2.append(symb)
                    # stack2.append(s)
                    # Found Cycle!
                    if _notOnlyEpsilon(symbolStack2) or _notOnlyEpsilonStack1(s):
                        return (s, symbolStack2, stack2)
                    symbolStack2.pop(-1)
                    # stack2.pop(-1)
                else:
                    if not s in exploredStatesCycle:
                        exploredStatesCycle.add(s)
                        stack2.append(s)
                        symbolStack2.append(symb)
                        res = _exploreDFSCycle(_successorStates(s))
                        if res is not None:
                            return res
                        symbolStack2.pop(-1)
                        stack2.pop(-1)

        def _exploreDFS(states):
            nonlocal stack2, stack1, symbolStack1, symbolStack2, exploredStates, exploredStatesCycle
            for (s, symb) in states:
                if not s in exploredStates:
                    exploredStates.add(s)
                    stack1.append(s)
                    symbolStack1.append(symb)
                    stack1Set.add(s)
                    if s in self._finalStates:
                        exploredStatesCycle = set()
                        res = _exploreDFSCycle(_successorStates(s))
                        if res is not None:
                            trans, cyclePref = _split(stack1, symbolStack1, s) 
                            cyclePost = res[1]
                            return (symbolStack1, cyclePref+cyclePost)
                    res = _exploreDFS(_successorStates(s))
                    if res is not None:
                        return res
                    stack1Set.remove(stack1.pop(-1))
                    symbolStack1.pop(-1)

        stack1 = []
        symbolStack1 = []
        stack1Set = set()
        stack2 = []
        symbolStack2 = []
        exploredStates = set()
        exploredStatesCycle = set()
        successorStates = { (s, self._epsilonSymbol) for s in self._initialStates}
        res = _exploreDFS(successorStates)
        if res is not None:
            # print("Cycle found!")            
            return (False, _stripEpsilon(res[0]), _stripEpsilon(res[1]))
        else:
            # print("No Cycle found")
            return (True, None, None)

    def languageEmptyBuchiAlternative(self):
        ''' Checks if the Buchi language is empty. Returns an accepted word and path if the language is not empty.'''

        # Step 1: find all reachable accepting states and determin a word sigma that leads to it

        # Step 2: For each accepting state s do:
        #   - find all states that can be reached with one non-epsilon symbol and remember that symbol a
        #   - check if s can be reached from any of those states by a word tau
        #   - (it may be OK to stop searching when we reaching any of the accpeting states we already covered)
        #   - if yes, the automaton accepts the word (sigma)(a tau)**

        # Step 1, determine the reachable accepting states
        reachableStates, words, paths = self.reachableStatesWithWordsAndPaths()
        reachableFinalStates = reachableStates.intersection(self._finalStates)

        # Step 2, check for each reachable accepting state if there is a cycle with a non-empty word that returns to it
        for s in sorted(reachableFinalStates):
            # to ensure that the cycle word is non-empty, first determine the states reachable from s with a non-epsilon symbol, and remember that symbol for each state 
            sClosure = self.epsilonClosure(set([s]))
            finalPlusSymbolReachableStates = set()
            singleSymbol = dict()
            for t in sorted(sClosure):
                for symb in sorted(self.outgoingSymbols(t)):
                    states = self.epsilonClosure(self.nextStates(t, symb))
                    finalPlusSymbolReachableStates.update(states)
                    for u in sorted(states):
                        singleSymbol[u] = [symb]

            # test if s is reachable from any state in finalPlusSymbolReachableStates
            cycleReachableStates, cycleWords, cyclePaths = self.reachableStatesWithWordsAndPaths(finalPlusSymbolReachableStates, singleSymbol)
            if s in cycleReachableStates:
                return (False, words[s], cycleWords[s], paths[s], cyclePaths[s])

        # print("No Cycle found")
        return (True, None, None, None, None)

    def languageIncluded(self, A):
        '''Check if the language of the automaton is included in the language of automaton A.
        If not, a word is returned that is in the language of the automaton, but not in the language of A'''
        AC = A.complement()
        P = AC.strictProduct(self)
        boolResult, word, _ = P.languageEmpty()
        return (boolResult, word)

    def subAutomaton(self, states):
        ''' return a subautomaton containing on the states in the set states '''

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
                for symb in self._transitions[s]:
                    for t in self._transitions[s][symb]:
                        if t in states:
                            result.addTransition(s, symb, t)

        return result

    def eliminateReachability(self):
        '''Reduce the size of the automaton by removing unreachable states and states from which no final state is reachable.'''

        # remove unreachable states
        states = self.reachableStates().intersection(self.reachableStatesFinal())
        return self.subAutomaton(states)

    def eliminateStatesWithoutOutgoingTransitions(self):
        toEliminate = set()
        for s in self._states:
            if len(self.outgoingSymbolsWithEpsilon(s)) ==0:
                toEliminate.add(s)

        if len(toEliminate) ==0:
            return self

        return self.subAutomaton(self._states.difference(toEliminate)).eliminateStatesWithoutOutgoingTransitions()

    def _partitionRefinement(self):

        def _createPartition(partitions, partitionMap, setOfStates):
            fSetOfStates = frozenset(setOfStates)
            partitions.add(fSetOfStates)
            for s in setOfStates:
                partitionMap[s] = fSetOfStates

        def _partitionRefinementEdgesEquivalent(s1, s2):

            # s1 and s2 are equivalent if for every s1-a->C, s2-a->C and vice versa
            labels = set()
            labels.update(self.outgoingSymbolsSet(self.epsilonClosure(set([s1]))))
            labels.update(self.outgoingSymbolsSet(self.epsilonClosure(set([s2]))))

            ecs1 = self.epsilonClosure(set([s1]))
            ecs2 = self.epsilonClosure(set([s2]))

            # for every label, compare outgong edges
            for l in labels:
                # collect classes of states in ns1 and ns2
                cs1 = set()
                cs2 = set()
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
        partitions = set()
        states_f = self._backwardEpsilonClosure(self._finalStates)
        states_nf = self._states.difference(states_f)
        if len(states_f):
            p_f = frozenset(states_f)
            partitions.add(p_f) 
        if len(states_nf):
            p_nf = frozenset(states_nf)
            partitions.add(p_nf)

        partitionMap = dict()
        for s in states_f:
            partitionMap[s] = p_f
        for s in states_nf:
            partitionMap[s] = p_nf

        oldPartitions = set()

        while len(oldPartitions) != len(partitions):
            # print(partitions)
            newPartitions = set()
            for clss in partitions:
                # pick arbitrary state from class
                s1 = next(iter(clss))
                # print('Selected state: {}'.format(s1))

                equivSet = set()
                remainingSet = set()
                equivSet.add(s1)

                # check whether all other states can go with the same labels to
                # the same set of other equivalence classes.
                for s2 in clss:
                    if s2 != s1:
                        if _partitionRefinementEdgesEquivalent(s1, s2):
                            equivSet.add(s2)
                        else:
                            remainingSet.add(s2)
                
                # if not, split the class
                if len(equivSet) == len(clss):
                    _createPartition(newPartitions, partitionMap, equivSet)
                else:
                    _createPartition(newPartitions, partitionMap, equivSet)
                    _createPartition(newPartitions, partitionMap, remainingSet)

            oldPartitions = partitions
            partitions = newPartitions

        return partitions, partitionMap

    def minimize(self):
        '''Implements a partition refinement strategy to reduce the size of the automaton.'''
        
        def setAsState(ss): return "{" + (",".join(sorted(ss))) + "}"


        # remove unreachable states
        # rmove states from which final states are not reachable
        interm = self.eliminateReachability()
        # find equalivalent state sthrough artition refinement.
        partitions, partitionMap = interm._partitionRefinement()


        result = Automaton()
 
        # make states
        for p in partitions:
            ns = setAsState(p)
            result.addState(ns)
            s = next(iter(p))
            if s in interm._backwardEpsilonClosure(interm._finalStates):
                result.makeFinalState(ns)

            # determine initial states
            # a partition is initial if one of its states was initial
            for s in p:
                if s in interm._initialStates:
                    result.makeInitialState(ns)


        # make transitions
        for p in partitions:
            # take a representative state
            s = next(iter(p))
            for t in interm.epsilonClosure(set([s])):
                for symb in interm.outgoingSymbols(t):
                    for u in interm.nextStates(t, symb):
                        result.addTransition(setAsState(partitionMap[s]), symb, setAsState(partitionMap[u]))
                if t in interm._epsilonTransitions:
                    for u in interm._epsilonTransitions[t]:
                        if partitionMap[s] != partitionMap[u]:
                            result.addEpsilonTransition(setAsState(partitionMap[s]), setAsState(partitionMap[u]))
        return result

    def minimizeBuchi(self):
        
        # eliminte states from which not all acceptance sets are reachable
        
        interm = self.eliminateStatesWithoutOutgoingTransitions()
        return  interm.minimize()

    def statesInBFSOrder(self):
        result = list()
        self._breadthFirstSearch(lambda s: result.append(s))
        return result
        
    def relabelStates(self):

        def _stateName(n):
            return "S{}".format(n)

        def _createState(s):
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

    def eliminateEpsilonTransitions(self):
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


    def _asDSLSymbol(self, symb):
        if re.match(r"[a-zA-Z][a-zA-Z0-9]*", symb):
            return symb
        return symb.replace('"', '\\"')

    def asDSL(self, name):

        def _addStateWithAttributes(u, statesOutput, output):
            output.write(u)
            if not u in statesOutput:
                self._dslOutputStateAttributes(u, output)
                statesOutput.add(u)

        # keep track of the states that have been output
        statesOutput = set()
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
                output.write(", ".join([self._asDSLSymbol(symb) for symb in sorted(symbols)]))
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
    def _ParsingAddStateWithLabels(fsa, s, labels, acceptanceSets):
        fsa.addState(s)
        for attr in labels:
            if attr == 'initial' or attr == 'i':
                fsa.makeInitialState(s)
            if attr == 'final' or attr == 'f':
                fsa.makeFinalState(s, acceptanceSets)

    @staticmethod
    def fromDSL(dslString):

        factory = dict()
        factory['Init'] = lambda : Automaton()
        factory['addTransitionPossiblyEpsilon'] = lambda fsa, s, t, symb: \
            fsa.addEpsilonTransition(s, t) if symb == Automaton._epsilonSymbol else fsa.addTransition(s, symb, t)
        factory['AddEpsilonTransition'] = lambda fsa, s, t : fsa.addEpsilonTransition(s, t)
        factory['AddState'] = lambda fsa, s, labels, acceptanceSets: Automaton._ParsingAddStateWithLabels(fsa, s, labels, acceptanceSets)
        result = parseFSADSL(dslString, factory)
        if result[0] is None:
            exit(1)
        return result
        
    def reachableStates(self):
        ''' return a set of all states reachable from an initial state '''
        result, _, _ = self.reachableStatesWithWordsAndPaths()
        return result

    def reachableStatesWithWordsAndPaths(self, startingStates=None, startingWords=None, startingPaths=None):
        ''' return a set of all states reachable from any state in startingStates and for each a word and a path by which is is reached. If startingStates is omitted, the initial states are used'''
        if startingStates is None:
            startingStates = sorted(self._initialStates)
        result = set()
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
                for symb in sorted(self._transitions[s]):
                    for t in sorted(self._transitions[s][symb]):
                        if not t in result:
                            statesToExplore.append(t)
                            words[t] = words[s] + [symb]
                            paths[t] = paths[s] + [t]

        return result, words, paths

    def reachableStatesFinal(self):
        ''' return the set of all states from which a final state is reachable '''
        result = set()
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
                for symb in self._transitions[t]:
                    for u in self._transitions[t][symb]:
                        if u == s:
                            if not t in result:
                                statesToExplore.add(t)
        return result

    def addGeneralizedBuchiAcceptanceSets(self, A):
        '''
        return a new on-generalized Büchi automaton with the added generalized acceptance sets incorporated
        '''
        res, _ = self.addGeneralizedBuchiAcceptanceSetsWithStateMap(A)
        return res

    def addGeneralizedBuchiAcceptanceSetsWithStateMap(self, A):
        '''
        return a new on-generalized Büchi automaton with the added generalized acceptance sets incorporated and a map linking the new states to the original states
        '''

        def _newState(s, n):
            return "({},F{})".format(s,str(n))
       
        stateMap = dict()

        # create a copy of every state for every acceptance set.
        # label final state accordingly
        # add transitions to state in same layer for non-accepting source states 
        # or state in next layer if it is accepting
        res = Automaton()
        acceptanceSets = []
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
                for symb in self._transitions[s]:
                    for t in self._transitions[s][symb]:
                        if (s in acceptanceSets[n]):
                            res.addTransition(_newState(s,n), symb, _newState(t,nxt))
                        else:
                            res.addTransition(_newState(s,n), symb, _newState(t,n))
            for s in self._epsilonTransitions:
                for t in self._epsilonTransitions[s]:
                    if (s in acceptanceSets[n]):
                        res.addEpsilonTransition(_newState(s,n), _newState(t,nxt))
                    else:
                        res.addEpsilonTransition(_newState(s,n), _newState(t,n))

        return res, stateMap

    def hasGeneralizedAcceptanceSets(self):
        return len(self._generalizedAcceptanceSets) > 0


    def asRegularBuchiAutomaton(self):
        res, _ = self.asRegularBuchiAutomatonWithStateMap()
        return res


    def asRegularBuchiAutomatonWithStateMap(self):
        if len(self._generalizedAcceptanceSets) == 0:
            stateMap = dict()
            for s in self._states:
                stateMap[s] = s
            return self, stateMap
        return self.addGeneralizedBuchiAcceptanceSetsWithStateMap(list(self._generalizedAcceptanceSets.values()))

	# collect common transitions into multi-labels
    def _dslMultiSymbolTransitions(self):
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
        result = set()
        for s in reorg.keys():
            for t in reorg[s].keys():
                result.add((s, frozenset(reorg[s][t]), t))
        return result

    def _dslOutputStateAttributes(self, state, output):
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

    def outgoingSymbols(self, state):
        if not state in self._transitions:
            return set()
        return frozenset(self._transitions[state].keys())

    def outgoingSymbolsSet(self, setOfStates):
        res = set()
        for s in setOfStates:
            res.update(self.outgoingSymbols(s))
        return frozenset(res)

    def outgoingSymbolsWithEpsilon(self, state):
        result = set(self.outgoingSymbols(state))
        if state in self._epsilonTransitions:
            result.add(Automaton._epsilonSymbol)
        return frozenset(result)

    def nextStates(self, state, symbol):
        """
            Return the set of states reachable from 'state' by a transition labelled 'symbol', where symbol is a non-epsilon symbol.
        """
        if not state in self._transitions:
            return set()
        if not symbol in self._transitions[state]:
            return set()
        return self._transitions[state][symbol]

    def nextStatesWithEpsilon(self, state, symbol):
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

    def nextEpsilonStates(self, state):
        """
            Return the set of states reachable from 'state' by a single epsilon transition.
        """
        if not state in self._epsilonTransitions:
            return set()
        return self._epsilonTransitions[state]

    def nextStatesEpsilonClosureWithPaths(self, state, symbol):
        """
            Return the set of states reachable from 'state' by a sequence of transitions including an arbitrary number of epsilon transitions and one 'symbol' transition, where symbol is not epsilon.
            Returns a tuple with:
            - a set of states thus reachable
            - a dictionary with for every reachable state s, a path starting in 'state' and ending in s.
        """
        return self.setNextStatesEpsilonClosureWithPaths(set([state]), symbol)

    def setNextStatesEpsilonClosureWithPaths(self, states, symbol):
        """
            Return the set of states reachable from a state from 'states' by a sequence of transitions including an arbitrary number of epsilon transitions and one 'symbol' transition, where symbol is not epsilon.
            Returns a tuple with:
            - a set of states thus reachable
            - a dictionary with for every reachable state s, a path starting in a state from 'states' and ending in s.
        """
        preEpsilonReachableStates, prePaths = self.epsilonClosureWithPaths(states)
        afterSymbolStates = set()
        afterSymbolPaths = dict()
        for s in preEpsilonReachableStates:
            for t in self.nextStates(s, symbol):
                afterSymbolStates.add(t)
                afterSymbolPaths[t] = prePaths[s] + [t]
        postEpsilonReachableStates, postPaths = self.epsilonClosureWithPaths(afterSymbolStates)
        resPaths = dict()
        for s in postEpsilonReachableStates:
            resPaths[s] = (afterSymbolPaths[postPaths[s][0]])[:-1] + postPaths[s]
        
        return postEpsilonReachableStates, resPaths


    def _setNextStates(self, states, symbol):
        nStates = set()
        for s in states:
            nStates.update(self.nextStates(s, symbol))
        return nStates

    def addStateUnique(self, state):
        # add a state named state, but ensure that it is unique by potentially modifying the name
        if not state in self._states:
            self.addState(state)
            return state
        n = 1
        while state+str(n) in self._states:
            n += 1
        newState = state+str(n)
        self.addState(newState)
        return newState

    def epsilonClosureWithPaths(self, setOfStates):
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
        return [res, paths]

    def epsilonClosure(self, setOfStates):
        res, _ = self.epsilonClosureWithPaths(setOfStates)
        return res

    def _backwardEpsilonClosure(self, setOfStates):
        """
            Determine the backward epsilon closure of the given set of states. Return a set of states reachable by zero or more epsilon transitions taken backward
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

    def _traceAcceptingWordAndPath(self, s, backTrack):
        word = []
        path = [s]
        t = s
        while not t in self._initialStates:
            (sym, t) = backTrack[t]
            if sym != self._epsilonSymbol:
                word.insert(0, sym)
            path.insert(0, t)
        return (False, word, path)

    def _successorStates(self, s):
        result = set()
        if s in self._transitions:
            for symb in self._transitions[s]:
                result.update(self.nextStates(s, symb))
        if s in self._epsilonTransitions:
            result.update(self._epsilonTransitions[s])
        return result

    def _breadthFirstSearch(self, visit):
        visitedStates = set()
        statesToVisit = sorted(list(self._initialStates))
        setOfStatesToVisit = set(self._initialStates)
        N=1
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

    def __str__(self):
        return "({}, {}, {}, {})".format(self._states, self._initialStates, self._finalStates, self._transitions)
