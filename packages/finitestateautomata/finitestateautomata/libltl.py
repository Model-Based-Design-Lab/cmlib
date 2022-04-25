from functools import reduce
from io import StringIO
from enum import Enum
import re

from finitestateautomata.libfsa import Automaton
from finitestateautomata.libltlgrammar import parseLTLDSL

LTLPROPOSITIONSIMPLE = re.compile(r"^[a-zA-Z]$")


def printUnfold(uf):
    print ("The set of pairs:")
    for p in uf:
        print("Now:")
        print(", ".join([str(phi) for phi in p[0]]))
        print("Next:")
        print(", ".join([str(phi) for phi in p[1]]))
        print("Accept:")
        print(", ".join([str(phi) for phi in p[2]]))



class LTLSubFormula(object):

    def __init__(self):
        pass

    def _bindingLevel(self):
        return 1

    def inNegationNormalForm(self):
        return self._inNegationNormalForm(False)

    def _inNegationNormalForm(self, propNeg):
        raise Exception("To be implemented in subclasses")

    def _inSetDNF(self):
        # default behavior, override where needed!
        conj = set()
        conj.add(self)
        result = set()
        result.add(frozenset(conj))
        return result

    def _unfold(self):
        # unfold in now and next
        # default behavior, overwrite when necessary!
        # returns a set (disjunction) of pairs now, next
        now = self._inSetDNF()
        return {(conj, frozenset([]), frozenset([])) for conj in now }

    def _setAcceptance(self, terms, a):
        return {(now, nxt, frozenset(acc.union(set([a])))) for (now, nxt, acc) in terms}


    def _getSubFormulas(self):
        # default behavior, override when necessary
        return set([self])
    
    def _localAlphabet(self):
        # default behavior, override when necessary
        return set()

    def _filterSymbols(self, symbols, propDefs):
        return symbols

    def alphabet(self):
        return reduce(lambda alph, phi: alph.union(phi._localAlphabet()), self._getSubFormulas(), set())

    def _isLivenessFormula(self):
        # default behavior, update if needed
        return False

    @staticmethod
    def _SetDNFAnd(s1, s2):
        result = set()
        for dt1 in s1:
            for dt2 in s2:
                nt = set()
                nt.update(dt1)
                nt.update(dt2)
                result.add(frozenset(nt))
        return result

    @staticmethod
    def _SetDNFOr(s1, s2):
        return s1.union(s2)

    @staticmethod
    def _pairSetDNFAnd(p1, p2):
        res = set()
        for dt1 in p1:
            for dt2 in p2:
                ntNow = set()
                ntNow.update(dt1[0])
                ntNow.update(dt2[0])
                ntNxt = set()
                ntNxt.update(dt1[1])
                ntNxt.update(dt2[1])
                ntAcc = set()
                ntAcc.update(dt1[2])
                ntAcc.update(dt2[2])
                res.add((frozenset(ntNow), frozenset(ntNxt), frozenset(ntAcc)))
        return res

class LTLFormulaTrue(LTLSubFormula):

    def __init__(self):
        pass

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        if propNeg:
            return LTLFormulaFalse()
        else: 
            return self

    def _inSetDNF(self):
        tt = frozenset()
        result = set()
        result.add(tt)
        return result

    def __str__(self):
        return "true"


class LTLFormulaFalse(LTLSubFormula):

    def __init__(self):
        pass

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        if propNeg:
            return LTLFormulaTrue()
        else: 
            return self

    def _inSetDNF(self):
        return set()

    def _filterSymbols(self, symbols, propDefs):
        return set()

    def __str__(self):
        return "false"

class LTLFormulaProposition(LTLSubFormula):

    def __init__(self, p, negated = False):
        self._proposition = p
        self._negated = negated

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        if propNeg:
            return LTLFormulaProposition(self._proposition, not self._negated)
        else: 
            return self
   
    def _localAlphabet(self):
        return set([self._proposition])

    def _filterSymbols(self, symbols, propDefs):
        if propDefs is None:
            propSymb = set([self._proposition])
        else:
            if self._proposition in propDefs:
                propSymb = propDefs[self._proposition]
            else:
                propSymb = set([self._proposition])
        
        if self._negated:
            return symbols.difference(propSymb)
        else:
            return symbols.intersection(propSymb)

    def __str__(self):
        if self._negated:
            pre = "not "
        else:
            pre = ""
        if LTLPROPOSITIONSIMPLE.match(self._proposition):
            return pre + self._proposition
        return pre + "'"+self._proposition.replace("'", "\\'")+"'"

class LTLFormulaUntil(LTLSubFormula):

    def __init__(self, phi1, phi2):
        self._phi1 = phi1
        self._phi2 = phi2

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        if propNeg:
            return LTLFormulaRelease(self._phi1._inNegationNormalForm(True), self._phi2._inNegationNormalForm(True))
        else: 
            return LTLFormulaUntil(self._phi1._inNegationNormalForm(False), self._phi2._inNegationNormalForm(False))

    def _unfold(self):
        # unfold in now and next
        # phi1 U phi2 = phi2 or (phi1 and X (phi1 U phi2))
        # rturn a set (disjunction) of pairs now, next

        uf2 = self._phi2._unfold()
        uf1 = self._phi1._unfold()
        nu = set([(frozenset([]), frozenset([self]), frozenset([self]))])
        return uf2.union(LTLSubFormula._pairSetDNFAnd(uf1, nu))

    def _getSubFormulas(self):
        # default behavior, override when necessary
        return set([self]).union(self._phi1._getSubFormulas()).union(self._phi2._getSubFormulas())

    def _isLivenessFormula(self):
        return True
   
    def __str__(self):
        return str(self._phi1) + "U" + str(self._phi2)

class LTLFormulaRelease(LTLSubFormula):

    def __init__(self, phi1, phi2):
        self._phi1 = phi1
        self._phi2 = phi2

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        if propNeg:
            return LTLFormulaUntil(self._phi1._inNegationNormalForm(True), self._phi2._inNegationNormalForm(True))
        else: 
            return LTLFormulaRelease(self._phi1._inNegationNormalForm(False), self._phi2._inNegationNormalForm(False))

    def _unfold(self):
        # unfold in now and next
        # phi1 R phi2 = phi2 and (phi1 or X (phi1 R phi2))
        # phi1 R phi2 = (phi2 and phi1) or (phi2 and X (phi1 R phi2)))
        # rturn a set (disjunction) of pairs now, next

        uf2 = self._phi2._unfold()
        uf1 = self._phi1._unfold()
        nr = set([(frozenset([]), frozenset([self]), frozenset([]))])
        alt1 = LTLSubFormula._pairSetDNFAnd(uf1, uf2)
        alt2 = LTLSubFormula._pairSetDNFAnd(uf2, nr)
        
        return alt1.union(alt2)

    def _getSubFormulas(self):
        # default behavior, override when necessary
        return set([self]).union(self._phi1._getSubFormulas()).union(self._phi2._getSubFormulas())

    def __str__(self):
        return str(self._phi1) + "R" + str(self._phi2)

class LTLFormulaImplication(LTLSubFormula):

    def __init__(self, phi1, phi2):
        self._phi1 = phi1
        self._phi2 = phi2

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        fs = set()
        fs.add(self._phi1._inNegationNormalForm(not propNeg))
        fs.add(self._phi2._inNegationNormalForm(propNeg))
        if propNeg:
            return LTLFormulaConjunction(fs)
        else: 
            return LTLFormulaDisjunction(fs)

    def _inSetDNF(self):
        raise Exception("remove implications first")

    def _getSubFormulas(self):
        # default behavior, override when necessary
        return set([self]).union(self._phi1._getSubFormulas()).union(self._phi2._getSubFormulas())

    def __str__(self):
        return str(self._phi1) + "=>" + str(self._phi1)

class LTLFormulaConjunction(LTLSubFormula):

    def __init__(self, subformulas):
        self._subformulas = subformulas

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        fs = {phi._inNegationNormalForm(propNeg) for phi in self._subformulas}
        if propNeg:
            return LTLFormulaDisjunction(fs)
        else: 
            return LTLFormulaConjunction(fs)

    def _inSetDNF(self):
        sub = [phi._inSetDNF() for phi in self._subformulas]
        result = LTLFormulaTrue()._inSetDNF()
        for s in sub:
            result = LTLSubFormula._SetDNFAnd(result, s)
        return result
    
    def _unfold(self):
        # unfold in now and next
        # rturn a set (disjunction) of pairs now, next

        res = set([(frozenset([]), frozenset([]), frozenset([]))])

        for phi in self._subformulas:
            res = LTLSubFormula._pairSetDNFAnd(res, phi._unfold())
        
        return res

    def _getSubFormulas(self):
        return reduce(lambda res, f: res.union(f._getSubFormulas()), self._subformulas, set())

    def __str__(self):
        return " and ".join([str(phi) for phi in self._subformulas])

class LTLFormulaDisjunction(LTLSubFormula):

    def __init__(self, subformulas):
        self._subformulas = subformulas

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        fs = {phi._inNegationNormalForm(propNeg) for phi in self._subformulas}
        if propNeg:
            return LTLFormulaConjunction(fs)
        else: 
            return LTLFormulaDisjunction(fs)

    def _inSetDNF(self):
        sub = [phi._inSetDNF() for phi in self._subformulas]
        result = LTLFormulaFalse()._inSetDNF()
        for s in sub:
            result = LTLSubFormula._SetDNFOr(result, s)
        return result

    def _unfold(self):
        # unfold in now and next
        # rturn a set (disjunction) of pairs now, next

        res = set()

        for phi in self._subformulas:
            res.update(phi._unfold())
        
        return res

    def _getSubFormulas(self):
        return reduce(lambda res, f: res.union(f._getSubFormulas()), self._subformulas, set())

    def __str__(self):
        return " or ".join([str(phi) for phi in self._subformulas])

class LTLFormulaNext(LTLSubFormula):

    def __init__(self, subformula):
        self._subformula = subformula

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        return LTLFormulaNext(self._subformula._inNegationNormalForm(propNeg))

    def _unfold(self):
        # unfold in now and next
        # X phi
        # rturn a set (disjunction) of pairs now, next

        nxt = set([(frozenset([]), frozenset([self._subformula]), frozenset([]))])     
        return nxt

    def _getSubFormulas(self):
        return set([self]).union(self._subformula._getSubFormulas())

    def __str__(self):
        return "X" + str(self._subformula)

class LTLFormulaNegation(LTLSubFormula):

    def __init__(self, subformula):
        self._subformula = subformula

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        return self._subformula._inNegationNormalForm(not propNeg)

    def _inSetDNF(self):
        raise Exception("remove implications first")

    def _getSubFormulas(self):
        return set([self]).union(self._subformula._getSubFormulas())

    def __str__(self):
        return "not" + str(self._subformula)

class LTLFormulaAlways(LTLSubFormula):

    def __init__(self, subformula):
        self._subformula = subformula

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        phi = self._subformula._inNegationNormalForm(propNeg)
        if propNeg:
            return LTLFormulaEventually(phi)
        else: 
            return LTLFormulaAlways(phi)

    def _unfold(self):
        # unfold in now and next
        # G phi = phi and XG phi
        # return a set (disjunction) of pairs now, next

        uf = self._subformula._unfold()
        ng = set([(frozenset([]), frozenset([self]), frozenset([]))])
        return LTLSubFormula._pairSetDNFAnd(uf, ng)

    def _getSubFormulas(self):
        return set([self]).union(self._subformula._getSubFormulas())

    def __str__(self):
        return "G" + str(self._subformula)

class LTLFormulaEventually(LTLSubFormula):

    def __init__(self, subformula):
        self._subformula = subformula

    def _bindingLevel(self):
        return 1

    def _inNegationNormalForm(self, propNeg):
        phi = self._subformula._inNegationNormalForm(propNeg)
        if propNeg:
            return LTLFormulaAlways(phi)
        else: 
            return LTLFormulaEventually(phi)

    def _unfold(self):
        # unfold in now and next
        # F phi = phi or XF phi
        # return a set (disjunction) of pairs now, next

        uf = self._subformula._unfold()
        nf = set([(frozenset([]), frozenset([self]), frozenset([self]))])   
        return uf.union(nf)

    def _getSubFormulas(self):
        return set([self]).union(self._subformula._getSubFormulas())

    def _isLivenessFormula(self):
        return True

    def __str__(self):
        return "F" + str(self._subformula)

class LTLFormula(object):

    def __init__(self, expression, alphabet, definitions):
        self._expression = expression
        self._alphabet = alphabet
        self._propDefinitions = definitions


    def _SetDNFAnd(s1, s2):
        result = set()
        for dt1 in s1:
            for dt2 in s2:
                nt = set()
                nt.update(dt1)
                nt.update(dt2)
                result.add(frozenset(nt))
        return result

    def _determineAlphabet(self):
        # if alphabet explicitly drefine return
        # other wise set of propositions, with those define replace by their sets,

        if self._alphabet is not None:
            return self._alphabet 
        propositions = self._expression.alphabet()
        if self._propDefinitions is None:
            return propositions
        res = set()
        for p in propositions:
            if p in self._propDefinitions:
                res.update(self._propDefinitions[p])
            else: res.add(p)
        return res



    def asFSA(self):

        def _unfold(s):
            # return a set pairs of (frozen)set of now and next formulas and acceptance sets
            result = set([(frozenset([]), frozenset([]), frozenset([]))])
            for phi in s:
                unf = phi._unfold()
                result = LTLSubFormula._pairSetDNFAnd(result, unf)
            # printUnfold(result)
            return result

        def stateString(s):
            return ','.join([str(f) for f in s])

        def printState(s):
            print(stateString(s))

        def printEdge(s):
            print(','.join([str(f) for f in s]))

        def addState(s, initial = False):
            nonlocal stateCounter
            ss = stateString(s)
            if ss in stateIndex:
                return False
            stateIndex[ss] = s
            stateName = "S"+str(stateCounter)
            states.add(stateName)
            if initial:
                initialStateNames.add(stateName)
            # F.addState(stateName)
            stateIndexF[ss] = stateName
            stateCounter+=1
            return True

        def _acceptanceIndex(s):
            return int(s[s.index("A")+1:s.index(")")])

        # partition transitions on acceptance sets
        # create a state for every incoming combination
        # determin acceptance sets
        # Let FSA reduce Generalized sets


        def addTransition(s, edgeSpec, t, acc):
            ss = stateString(s)
            ts = stateString(t)

            symbols = reduce(lambda res, f: f._filterSymbols(res, self._propDefinitions), edgeSpec, alphabet)
            for symb in symbols:
                trans = (stateIndexF[ss], symb, stateIndexF[ts])
                acceptanceSets[trans] = set()
                acceptanceSets[trans].update({str(a) for a in acc})

        livenessFormulas = [f for f in self._expression._getSubFormulas() if f._isLivenessFormula()]
        # acceptanceSets = dict([(str(f), set()) for f in livenessFormulas])
        acceptanceSets = dict()

        initialStates = self._expression.inNegationNormalForm()._inSetDNF()
        initialStateNames = set()
        statesToUnfold = initialStates.copy()

        # check if we have an explicit alphabet, otherwise, compute
        alphabet = self._determineAlphabet()

        F = Automaton()
        stateCounter = 1
        stateIndexF = dict()


        states = set()
        stateIndex = dict([(stateString(s), s) for s in states])

        for s in statesToUnfold:
            addState(s, True)

        while len(statesToUnfold) > 0:
            # take one state from the set
            s = next(iter(statesToUnfold))
            statesToUnfold.remove(s)
            
            # determine transitions
            transitions = _unfold(s)
            for t in transitions:
                if addState(t[1]):
                    statesToUnfold.add(t[1])
                addTransition(s, t[0], t[1], t[2])

        # for all state collect the incoming acceptance labels
        # and store them in a dictionary
        acceptanceSetsStates = dict([(s, set()) for s in states])
        # for all transitions, keys of acceptanceSets
        for t in acceptanceSets:
            # add to the target state
            acceptanceSetsStates[t[2]].add(frozenset(acceptanceSets[t]))
        
        # collect all the labels from all states
        acceptanceLabels = reduce(lambda res, acc: res.union(acc), acceptanceSetsStates.values(), set())
        # print(acceptanceLabels)
        
        # associate all acceptance labels with a unique number starting from 1
        stateLabelsAcceptance = dict()
        k = 1
        for a in acceptanceLabels:
            if len(a) > 0:
                stateLabelsAcceptance[a] = k
                k += 1

        # ultimate state names are formed as a combination of state name and acceptance label number
        buildStateName = lambda s, n: "({},A{})".format(s, str(n))

        # add states without acceptance label
        for s in states:
            F.addState(buildStateName(s, 0))

        # make initial states
        for s in initialStateNames:
            F.makeInitialState(buildStateName(s, 0))

        # for all states with incoming acceptance labels make different states
        for s in acceptanceSetsStates:
            for a in acceptanceSetsStates[s]:
                if len(a) > 0:
                    F.addState(buildStateName(s, stateLabelsAcceptance[a]))

        # construct appropriate transitions
        for t in acceptanceSets:
            a = frozenset(acceptanceSets[t])
            if len(a) == 0:
                targetState = buildStateName(t[2], 0)
            else:
                targetState = buildStateName(t[2], stateLabelsAcceptance[a])
            for s in F.states():
                if s.startswith("("+t[0]):
                    F.addTransition(s, t[1], targetState)

        # compute acceptance sets
        # one for each non-empy se
        acceptanceSetsKeys = reduce(lambda res, acc: res.union(acc), acceptanceLabels, set())
        generalizedAcceptanceSets = []
        for k in acceptanceSetsKeys:
            acceptanceStateLabels = [stateLabelsAcceptance[l] for l in acceptanceLabels if (k in l)]
            accSetOfStates = [s for s in F.states() if _acceptanceIndex(s) in acceptanceStateLabels]
            generalizedAcceptanceSets.append(frozenset(F.states().difference(accSetOfStates)))

        # if there are no acceptance sets, make all states accepting
        if len(generalizedAcceptanceSets) == 0:
            for s in F.states():
                F.makeFinalState(s)
        else:
            # set the first acceptance set as accepting states
            for s in generalizedAcceptanceSets[0]:
                F.makeFinalState(s)
            # add all following ones as trnasformations
            F =F.addGeneralizedBuchiAcceptanceSets(generalizedAcceptanceSets[1:])

        return F


    def asDSL(self, name):
        return 'ltl formula {} = {}'.format(name, str(self))

    @staticmethod
    def fromDSL(regexString):
        factory = dict()
        factory['Proposition'] = lambda p: LTLFormulaProposition(p)
        factory['Until'] = lambda l, r: LTLFormulaUntil(l, r)
        factory['Release'] = lambda l, r: LTLFormulaRelease(l, r)
        factory['Next'] = lambda f: LTLFormulaNext(f)
        factory['Always'] = lambda f: LTLFormulaAlways(f)
        factory['Negation'] = lambda f: LTLFormulaNegation(f)
        factory['Eventually'] = lambda f: LTLFormulaEventually(f)
        factory['Disjunction'] = lambda fs: LTLFormulaDisjunction(fs)
        factory['Conjunction'] = lambda fs: LTLFormulaConjunction(fs)
        factory['True'] = lambda : LTLFormulaTrue()
        factory['False'] = lambda : LTLFormulaFalse()
        (name, form) = parseLTLDSL(regexString, factory)
        if form is None:
            exit(1)
        (phi, alphabet, definitions) = form
        return name, LTLFormula(phi, alphabet, definitions)


    def __str__(self):
        return str(self._expression)
