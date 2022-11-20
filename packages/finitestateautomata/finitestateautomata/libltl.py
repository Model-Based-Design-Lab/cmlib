from functools import reduce
import re
from typing import AbstractSet, Callable, Dict, Optional, Set, Tuple

from finitestateautomata.libfsa import Automaton
from finitestateautomata.libltlgrammar import parseLTLDSL

LTLPROPOSITIONSIMPLE = re.compile(r"^[a-zA-Z]$")


TDisjunctiveNormalForm = Set[AbstractSet['LTLSubFormula']]
TConjunctiveNormalForm = AbstractSet['LTLSubFormula']


def printUnfold(uf:Set[Tuple[AbstractSet['LTLFormula'],AbstractSet['LTLFormula'],AbstractSet['LTLFormula']]]):
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

    def inNegationNormalForm(self)->'LTLSubFormula':
        '''Determine the equivalent formula in negation normal form, i.e., having negations only in front of propositions. It does not always create a new formula object.'''
        return self._inNegationNormalForm(False)

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        '''Determine the equivalent formula in negation normal form, i.e., having negations only in front of propositions. propNeg indicates on a negation is being propagated or not.'''
        raise Exception("To be implemented in subclasses")

    def _inSetDNF(self)->TDisjunctiveNormalForm:
        '''Return formula in disjunctive normal form as a set (disjunction) of sets (conjunction) of formulas.'''
        # default behavior, override where needed!
        conj: Set['LTLSubFormula'] = set()
        conj.add(self)
        result: AbstractSet[AbstractSet[LTLSubFormula]] = set()
        result.add(frozenset(conj))
        return result

    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        ''' Unfold formula into a set (disjunctive) of 3-tuples consisting of a set (conjunctive) of 'now' formulas, a set (conjunctive) of 'next' formulas and a set of acceptance sets, i.e., until or eventually formulas whose eventualities are satisfied in that disjunctive term. '''
        # default behavior, keep all terms in now. Overwrite when necessary!
        # determine the DNF
        now = self._inSetDNF()
        # return set of triples
        return {(conj, frozenset([]), frozenset([])) for conj in now }

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        '''Return the set of all subformulas.'''
        # default behavior, override when necessary
        return set([self])
    
    def _localAlphabet(self)->Set[str]:
        '''Return the set of atomic propositions of the formula. '''
        # default behavior, override when necessary
        return set()

    def _filterSymbols(self, symbols: Set[str], propDefs:Dict[str,Set[str]])->Set[str]:
        '''Return symbols that satisfy the propositional formula.'''
        # default all, override if necessary
        return symbols

    def alphabet(self)->Set[str]:
        '''Return the set of propositions in the formula.'''
        return reduce(lambda alpha, phi: alpha.union(phi._localAlphabet()), self._getSubFormulas(), set())

    def _isLivenessFormula(self)->bool:
        '''Return if this is a formula representing a liveness constraint.'''
        # default behavior, update if needed
        return False

    @staticmethod
    def _SetDNFAnd(s1: TDisjunctiveNormalForm, s2: TDisjunctiveNormalForm) -> TDisjunctiveNormalForm:
        '''Perform logical and operation on two formulas in set disjunctive normal form.'''
        result:TDisjunctiveNormalForm = set()
        for dt1 in s1:
            for dt2 in s2:
                nt: TConjunctiveNormalForm = set()
                nt.update(dt1)
                nt.update(dt2)
                result.add(frozenset(nt))
        return result

    @staticmethod
    def _SetDNFOr(s1: TDisjunctiveNormalForm, s2: TDisjunctiveNormalForm) -> TDisjunctiveNormalForm:
        '''Perform logical or operation on two formulas in set disjunctive normal form.'''
        return s1.union(s2)

    @staticmethod
    def _pairSetDNFAnd(
        p1: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]], 
        p2: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]) -> Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        '''Perform logical and operation on now, next, acceptance set triples'''
        res = set()
        for dt1 in p1:
            for dt2 in p2:
                ntNow: TConjunctiveNormalForm = set()
                ntNow.update(dt1[0])
                ntNow.update(dt2[0])
                ntNxt: TConjunctiveNormalForm = set()
                ntNxt.update(dt1[1])
                ntNxt.update(dt2[1])
                ntAcc:Set['LTLSubFormula'] = set()
                ntAcc.update(dt1[2])
                ntAcc.update(dt2[2])
                res.add((frozenset(ntNow), frozenset(ntNxt), frozenset(ntAcc)))
        return res

class LTLFormulaTrue(LTLSubFormula):
    '''Formula true'''
    
    def __init__(self):
        pass

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        if propNeg:
            return LTLFormulaFalse()
        else: 
            return self

    def _inSetDNF(self)->TDisjunctiveNormalForm:
        tt: TConjunctiveNormalForm = frozenset()
        result: TDisjunctiveNormalForm = set()
        result.add(tt)
        return result

    def __str__(self):
        return "true"


class LTLFormulaFalse(LTLSubFormula):

    def __init__(self):
        pass

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        if propNeg:
            return LTLFormulaTrue()
        else: 
            return self

    def _inSetDNF(self)->TDisjunctiveNormalForm:
        return set()

    def _filterSymbols(self, symbols: Set[str], propDefs:Dict[str,Set[str]])->Set[str]:
        return set()

    def __str__(self):
        return "false"

class LTLFormulaProposition(LTLSubFormula):

    _proposition: str
    _negated: bool

    def __init__(self, p: str, negated:bool = False):
        self._proposition = p
        self._negated = negated

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        if propNeg:
            return LTLFormulaProposition(self._proposition, not self._negated)
        else: 
            return self
   
    def _localAlphabet(self)->Set[str]:
        return set([self._proposition])

    def _filterSymbols(self, symbols: Set[str], propDefs:Dict[str,Set[str]])->Set[str]:
        if propDefs is None:
            propSymbols = set([self._proposition])
        else:
            if self._proposition in propDefs:
                propSymbols = propDefs[self._proposition]
            else:
                propSymbols = set([self._proposition])
        
        if self._negated:
            return symbols.difference(propSymbols)
        else:
            return symbols.intersection(propSymbols)

    def __str__(self):
        if self._negated:
            pre = "not "
        else:
            pre = ""
        if LTLPROPOSITIONSIMPLE.match(self._proposition):
            return pre + self._proposition
        return pre + "'"+self._proposition.replace("'", "\\'")+"'"

class LTLFormulaUntil(LTLSubFormula):

    _phi1: LTLSubFormula
    _phi2: LTLSubFormula

    def __init__(self, phi1: LTLSubFormula, phi2: LTLSubFormula):
        self._phi1 = phi1
        self._phi2 = phi2

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        if propNeg:
            # negation converts into Release formula
            return LTLFormulaRelease(self._phi1._inNegationNormalForm(True), self._phi2._inNegationNormalForm(True))
        else: 
            return LTLFormulaUntil(self._phi1._inNegationNormalForm(False), self._phi2._inNegationNormalForm(False))

    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        # unfold in now and next using the following identity
        # phi1 U phi2 = phi2 or (phi1 and X (phi1 U phi2))
        # return a set (disjunction) of triples now, next, acceptance set

        # unfold phi2
        uf2: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]] 
        uf2 = self._phi2._unfold()  # type: ignore couldn't make the type checker happy

        # unfold phi1        
        uf1: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]
        uf1 = self._phi1._unfold()  # type: ignore couldn't make the type checker happy
        
        # create the next part,  X (phi1 U phi2)
        # note that until formulas create acceptance sets in the equivalent automata
        nu: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet[LTLSubFormula]]] = set([(frozenset([]), frozenset([self]), frozenset([self]))])
        
        # return the disjunction (by set union) of uf2  and the conjunction (_pairSetDNFAnd) of uf1 and nu
        return uf2.union(LTLSubFormula._pairSetDNFAnd(uf1, nu))

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        # self, and recursively the subformulas of phi1 and ph2
        return set([self]).union(self._phi1._getSubFormulas()).union(self._phi2._getSubFormulas())

    def _isLivenessFormula(self):
        return True
   
    def __str__(self):
        return str(self._phi1) + "U" + str(self._phi2)

class LTLFormulaRelease(LTLSubFormula):

    _phi1: LTLSubFormula
    _phi2: LTLSubFormula

    def __init__(self, phi1: LTLSubFormula, phi2: LTLSubFormula):
        self._phi1 = phi1
        self._phi2 = phi2

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        if propNeg:
            # the negation of a Release formula is an Until formula
            return LTLFormulaUntil(self._phi1._inNegationNormalForm(True), self._phi2._inNegationNormalForm(True))
        else: 
            return LTLFormulaRelease(self._phi1._inNegationNormalForm(False), self._phi2._inNegationNormalForm(False))

    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # phi1 R phi2 = phi2 and (phi1 or X (phi1 R phi2))
        # phi1 R phi2 = (phi1 and phi2) or (phi2 and X (phi1 R phi2)))
        # return a set (disjunction) of triples now, next, acceptance / eventualities

        uf2: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]] 
        uf2 = self._phi2._unfold()  # type: ignore couldn't make the type checker happy
        uf1: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]] 
        uf1 = self._phi1._unfold()  # type: ignore couldn't make the type checker happy
        nr: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet[LTLSubFormula]]] = set([(frozenset([]), frozenset([self]), frozenset([]))])
        
        # alternative 1: phi1 and phi2
        alt1 = LTLSubFormula._pairSetDNFAnd(uf1, uf2)
        # alternative 2: phi2 and X (phi1 R phi2)
        alt2 = LTLSubFormula._pairSetDNFAnd(uf2, nr)
        
        return alt1.union(alt2)

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._phi1._getSubFormulas()).union(self._phi2._getSubFormulas())

    def __str__(self):
        return str(self._phi1) + "R" + str(self._phi2)

class LTLFormulaImplication(LTLSubFormula):

    _phi1: LTLSubFormula
    _phi2: LTLSubFormula

    def __init__(self, phi1: LTLSubFormula, phi2: LTLSubFormula):
        self._phi1 = phi1
        self._phi2 = phi2

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        # use the identity that phi1 => phi2 = not phi1 or phi2        
        fs = set()
        fs.add(self._phi1._inNegationNormalForm(not propNeg))
        fs.add(self._phi2._inNegationNormalForm(propNeg))
        if propNeg:
            return LTLFormulaConjunction(fs)
        else: 
            return LTLFormulaDisjunction(fs)

    def _inSetDNF(self)->TDisjunctiveNormalForm:
        "Not implemented. Implication is rewritten to disjunction before using this function."
        raise Exception("remove implications first")

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._phi1._getSubFormulas()).union(self._phi2._getSubFormulas())

    def __str__(self):
        return str(self._phi1) + "=>" + str(self._phi1)

class LTLFormulaConjunction(LTLSubFormula):

    '''Conjunction of a set (not necessarily two) of subformulas'''

    _subformulas: Set[LTLSubFormula]

    def __init__(self, subformulas: Set[LTLSubFormula]):
        self._subformulas = subformulas

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        fs = {phi._inNegationNormalForm(propNeg) for phi in self._subformulas}
        if propNeg:
            return LTLFormulaDisjunction(fs)
        else: 
            return LTLFormulaConjunction(fs)

    def _inSetDNF(self)->TDisjunctiveNormalForm:
        sub = [phi._inSetDNF() for phi in self._subformulas]
        result:TDisjunctiveNormalForm = LTLFormulaTrue()._inSetDNF()
        for s in sub:
            result = LTLSubFormula._SetDNFAnd(result, s)
        return result
    
    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # return a set (disjunction) of pairs now, next

        res:AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]] = set([(frozenset([]), frozenset([]), frozenset([]))])

        for phi in self._subformulas:
            unfoldedPhi: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]
            unfoldedPhi = phi._unfold()  # type: ignore
            res = LTLSubFormula._pairSetDNFAnd(res, unfoldedPhi)
        
        return res

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        l: Callable[[Set[LTLSubFormula], LTLSubFormula], Set[LTLSubFormula]] = lambda res, f: res.union(f._getSubFormulas())
        return reduce(l, self._subformulas, set())

    def __str__(self):
        return " and ".join([str(phi) for phi in self._subformulas])

class LTLFormulaDisjunction(LTLSubFormula):

    _subformulas: Set[LTLSubFormula]
    
    def __init__(self, subformulas: Set[LTLSubFormula]):
        self._subformulas = subformulas

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        fs = {phi._inNegationNormalForm(propNeg) for phi in self._subformulas}
        if propNeg:
            return LTLFormulaConjunction(fs)
        else: 
            return LTLFormulaDisjunction(fs)

    def _inSetDNF(self)->TDisjunctiveNormalForm:
        sub = [phi._inSetDNF() for phi in self._subformulas]
        result = LTLFormulaFalse()._inSetDNF()
        for s in sub:
            result = LTLSubFormula._SetDNFOr(result, s)
        return result

    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # return a set (disjunction) of pairs now, next

        res = set()

        for phi in self._subformulas:
            res.update(phi._unfold())
        
        return res

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:

        l: Callable[[Set[LTLSubFormula], LTLSubFormula], Set[LTLSubFormula]] = lambda res, f: res.union(f._getSubFormulas())
        return reduce(l, self._subformulas, set())

    def __str__(self):
        return " or ".join([str(phi) for phi in self._subformulas])

class LTLFormulaNext(LTLSubFormula):

    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        return LTLFormulaNext(self._subformula._inNegationNormalForm(propNeg))

    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # X phi

        nxt = set([(frozenset([]), frozenset([self._subformula]), frozenset([]))])     
        return nxt

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula._getSubFormulas())

    def __str__(self):
        return "X" + str(self._subformula)

class LTLFormulaNegation(LTLSubFormula):
    
    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        return self._subformula._inNegationNormalForm(not propNeg)

    def _inSetDNF(self)->TDisjunctiveNormalForm:
        # not implemented, assumes the formula is transformed to negation normal form first.
        raise Exception("Transform to negation normal form first")

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula._getSubFormulas())

    def __str__(self):
        return "not" + str(self._subformula)

class LTLFormulaAlways(LTLSubFormula):
    
    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        phi = self._subformula._inNegationNormalForm(propNeg)
        if propNeg:
            return LTLFormulaEventually(phi)
        else: 
            return LTLFormulaAlways(phi)

    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # G phi = phi and XG phi
        
        uf:Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]
        uf = self._subformula._unfold()  # type: ignore
        ng: AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]] = set([(frozenset([]), frozenset([self]), frozenset([]))])
        return LTLSubFormula._pairSetDNFAnd(uf, ng)

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula._getSubFormulas())

    def __str__(self):
        return "G" + str(self._subformula)

class LTLFormulaEventually(LTLSubFormula):

    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def _inNegationNormalForm(self, propNeg: bool)->'LTLSubFormula':
        phi = self._subformula._inNegationNormalForm(propNeg)
        if propNeg:
            return LTLFormulaAlways(phi)
        else: 
            return LTLFormulaEventually(phi)

    def _unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # F phi = phi or XF phi

        uf = self._subformula._unfold()
        nf: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]
        nf = set([(frozenset([]), frozenset([self]), frozenset([self]))])   
        return uf.union(nf)  # type: ignore

    def _getSubFormulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula._getSubFormulas())

    def _isLivenessFormula(self):
        return True

    def __str__(self):
        return "F" + str(self._subformula)

class LTLFormula(object):

    _expression: LTLSubFormula
    _alphabet: Optional[Set[str]]
    _propDefinitions: Dict[str,Set[str]]

    def __init__(self, expression: LTLSubFormula, alphabet: Optional[Set[str]], definitions: Optional[Dict[str,Set[str]]]):
        self._expression = expression
        self._alphabet = alphabet
        self._propDefinitions = definitions if definitions is not None else dict()

    def _SetDNFAnd(self, s1: TDisjunctiveNormalForm, s2: TDisjunctiveNormalForm) -> TDisjunctiveNormalForm:
        '''Determine the conjunction of two formulas in set-disjunctive normal form, i.e., a set (disjunction) of sets (conjunctions) of formulas.'''
        result = set()
        for dt1 in s1:
            for dt2 in s2:
                nt = set()
                nt.update(dt1)
                nt.update(dt2)
                result.add(frozenset(nt))
        return result

    def _determineAlphabet(self)->Set[str]:
        # if alphabet is explicitly defined return it
        # other wise set of propositions, with those define replace by their sets,

        if self._alphabet is not None:
            return self._alphabet 
        # get the propositions from the formula expression
        propositions = self._expression.alphabet()
        # if there are no other definitions, return the propositions from the formula
        if self._propDefinitions is None:
            return propositions
        
        # determine the alphabet from the defined propositions
        res: Set[str] = set()
        for p in propositions:
            if p in self._propDefinitions:
                res.update(self._propDefinitions[p])
            else: res.add(p)
        return res

    def asFSA(self)->Automaton:
        '''Convert the LTL formula to a Büchi automaton that accepts precisely all the words that satisfy the formula.'''

        def _unfold(s: AbstractSet[LTLSubFormula])->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]:
            '''Unfold set of subformulas into DNF and splitting now and next.'''
            
            # return a set of triples of (frozen) sets of now and next formulas and acceptance sets
            result:AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]] = set([(frozenset([]), frozenset([]), frozenset([]))])
            for phi in s:
                unf: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]
                unf = phi._unfold()  # type: ignore
                result = LTLSubFormula._pairSetDNFAnd(result, unf)
            return result

        def stateString(s: AbstractSet[LTLSubFormula])->str:
            return ','.join([str(f) for f in s])

        def printState(s):
            print(stateString(s))

        def printEdge(s):
            print(','.join([str(f) for f in s]))

        # set of states by names
        states: Set[str]
        # index of the sets of subformulas corresponding to the states
        stateIndex: Dict[str,AbstractSet[LTLSubFormula]]
        # a counter of how many states we have created
        stateCounter: int
        # names of the initial states
        initialStateNames: Set[str]
        # lookup from set of formulas (as a string) to the name of the corresponding state
        stateIndexF: Dict[str,str]
        # keep track of the set of acceptance sets associated with the transitions
        acceptanceSets: Dict[Tuple[str,str,str],Set[str]]

        def addState(s: AbstractSet[LTLSubFormula], initial: bool = False)->bool:
            '''Add a state. Returns True if the state was added, False if it already exists.'''
            
            nonlocal stateCounter
            nonlocal states
            nonlocal stateIndex
            nonlocal initialStateNames
            nonlocal stateIndexF

            # determine string representation for the state
            ss: str = stateString(s)
            # check if it already exists in the state index
            if ss in stateIndex:
                return False
        
            # add it to the index
            stateIndex[ss] = s
            # generate a new name for the state
            stateName = "S"+str(stateCounter)
            # update counter
            stateCounter+=1
            # add the state
            states.add(stateName)
            if initial:
                initialStateNames.add(stateName)
            stateIndexF[ss] = stateName
            return True

        def _acceptanceIndex(s: str)->int:
            '''Get index of acceptance set from state name'''
            return int(s[s.index("A")+1:s.index(")")])

        # partition transitions on acceptance sets
        # create a state for every incoming combination
        # determine acceptance sets
        # Let FSA reduce Generalized sets

        def addTransition(s: AbstractSet[LTLSubFormula], edgePropositionalFormula: AbstractSet[LTLSubFormula], t: AbstractSet[LTLSubFormula], acc: AbstractSet[LTLSubFormula]):
            '''Add a transition from the state corresponding to s, to the state corresponding to t, edges labelled by the propositional formula edgeSpec and by acceptance set acc'''
            
            nonlocal alphabet

            # get the corresponding strings
            ss: str = stateString(s)
            ts: str = stateString(t)

            # determine the symbols of the alphabet that match the propositional formula on the edge
            symbols = reduce(lambda res, f: f._filterSymbols(res, self._propDefinitions), edgePropositionalFormula, alphabet)
            for symbol in symbols:
                # create an edge from the state corresponding to s to the state corresponding to t, labelled with symbol
                trans = (stateIndexF[ss], symbol, stateIndexF[ts])
                # create an entry in the acceptanceSets dictionary
                acceptanceSets[trans] = set()
                # add the acceptance sets (as strings) from acc
                acceptanceSets[trans].update({str(a) for a in acc})

        # initialize
        acceptanceSets = dict()
        # initial states from the initial formula expression
        initialStates = self._expression.inNegationNormalForm()._inSetDNF()
        initialStateNames = set()
        # statesToUnfold keeps track of newly created states that need to be unfolded into now and next, initialized with the initial states
        statesToUnfold: Set[AbstractSet[LTLSubFormula]] = initialStates.copy()  # type: ignore

        # check if we have an explicit alphabet, otherwise, compute
        alphabet = self._determineAlphabet()

        # Create the automaton
        stateCounter = 1
        stateIndexF = dict()
        states = set()
        stateIndex = dict()

        # add the states we start from as initial states
        for s in statesToUnfold:
            addState(s, True)

        # as long as we states that still need to be unfolded
        while len(statesToUnfold) > 0:
            # take one state from the set and remove it
            s = next(iter(statesToUnfold))
            statesToUnfold.remove(s)
            
            # determine outgoing transitions from unfolded state
            transitions = _unfold(s)
            for t in transitions:
                t1: Set[LTLSubFormula] = t[1]  # type: ignore
                # add state if it doesn't exist yet
                if addState(t1):
                    # the state is new, so needs unfolding
                    statesToUnfold.add(t1)
                # add a transition from s to the new state with appropriate labels of propositional formula and acceptance sets
                addTransition(s, t[0], t[1], t[2])

        # for all states collect the incoming acceptance labels
        # and store them in a dictionary
        # create initial dictionary with empty sets
        # maps a state (str) to a set of set of str
        acceptanceSetsStates: Dict[str,Set[AbstractSet[str]]]
        acceptanceSetsStates = dict([(s, set()) for s in states])
        # for all transitions, keys of acceptanceSets
        for t in acceptanceSets:
            # add the acceptance set to the set of acceptance sets of the target state
            acceptanceSetsStates[t[2]].add(frozenset(acceptanceSets[t]))
        
        # collect all the acceptance labels from all states
        acceptanceLabels: Set[AbstractSet[str]]
        acceptanceLabels = reduce(lambda res, acc: res.union(acc), acceptanceSetsStates.values(), set())
        # print(acceptanceLabels)
        
        # associate all (non empty) acceptance labels with a unique number starting from 1
        stateLabelsAcceptance: Dict[AbstractSet[str],int] = dict()
        k = 1
        for a in acceptanceLabels:
            if len(a) > 0:
                stateLabelsAcceptance[a] = k
                k += 1

        # ultimate state names are formed as a combination of state name and acceptance label number
        buildStateName = lambda s, n: "({},A{})".format(s, str(n))

        F = Automaton()

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
        # one for each non-empty set
        acceptanceSetsKeys = reduce(lambda res, acc: res.union(acc), acceptanceLabels, set())
        generalizedAcceptanceSets = []
        for k in acceptanceSetsKeys:
            acceptanceStateLabels = [stateLabelsAcceptance[l] for l in acceptanceLabels if (k in l)]
            accSetOfStates = [s for s in F.states() if _acceptanceIndex(s) in acceptanceStateLabels]
            generalizedAcceptanceSets.append(frozenset(F.states().difference(accSetOfStates)))

        # if there are no acceptance sets, we need to make all states accepting
        if len(generalizedAcceptanceSets) == 0:
            for s in F.states():
                F.makeFinalState(s)
        else:
            # set the first acceptance set as accepting states
            for s in generalizedAcceptanceSets[0]:
                F.makeFinalState(s)
            # add all following ones as transformations
            F = F.addGeneralizedBuchiAcceptanceSets(generalizedAcceptanceSets[1:])

        return F


    def asDSL(self, name):
        return 'ltl formula {} = {}'.format(name, str(self))

    @staticmethod
    def fromDSL(ltlString)->Tuple[str,'LTLFormula']:
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
        (name, form) = parseLTLDSL(ltlString, factory)
        if form is None or name is None:
            exit(1)
        phi: LTLSubFormula
        alphabet: Optional[Set[str]]
        definitions: Optional[Dict[str,Set[str]]]
        (phi, alphabet, definitions) = form
        return name, LTLFormula(phi, alphabet, definitions)

    def __str__(self)->str:
        return str(self._expression)
