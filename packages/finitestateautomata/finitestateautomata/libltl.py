'''Library of analysis of Linear Temporal Logic.'''

from functools import reduce
import re
import sys
from typing import AbstractSet, Callable, Dict, Optional, Set, Tuple

from sortedcontainers import SortedDict, SortedSet

from finitestateautomata.libfsa import Automaton
from finitestateautomata.libltlgrammar import parse_ltl_dsl
from finitestateautomata.utils.utils import FiniteStateAutomataException

LTL_PROPOSITION_SIMPLE = re.compile(r"^[a-zA-Z]$")

TDisjunctiveNormalForm = Set[AbstractSet['LTLSubFormula']]
TConjunctiveNormalForm = AbstractSet['LTLSubFormula']

class LTLException(FiniteStateAutomataException):
    ''' Exceptions related to LTL '''

def print_unfold(uf:Set[Tuple[AbstractSet['LTLFormula'],AbstractSet['LTLFormula'], \
                              AbstractSet['LTLFormula']]]):
    '''Print unfolded formula.'''
    print ("The set of pairs:")
    for p in uf:
        print("Now:")
        print(", ".join([str(phi) for phi in p[0]]))
        print("Next:")
        print(", ".join([str(phi) for phi in p[1]]))
        print("Accept:")
        print(", ".join([str(phi) for phi in p[2]]))

class LTLSubFormula:
    '''Representation of an LTL formula.'''

    def __init__(self):
        pass

    def in_negation_normal_form(self)->'LTLSubFormula':
        '''Determine the equivalent formula in negation normal form, i.e., having negations
        only in front of propositions. It does not always create a new formula object.'''
        return self.in_negation_normal_form_prop(False)

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        '''Determine the equivalent formula in negation normal form, i.e., having negations
        only in front of propositions. prop_neg indicates on a negation is being propagated
        or not.'''
        raise LTLException("To be implemented in subclasses")

    def in_set_dnf(self)->TDisjunctiveNormalForm:
        '''Return formula in disjunctive normal form as a set (disjunction) of
        sets (conjunction) of formulas.'''

        # default behavior, override where needed!
        conj: Set['LTLSubFormula'] = set()
        conj.add(self)
        result: AbstractSet[AbstractSet[LTLSubFormula]] = set()
        result.add(frozenset(conj))
        return result

    def unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
                                         AbstractSet['LTLSubFormula']]]:
        ''' Unfold formula into a set (disjunctive) of 3-tuples consisting of a set
        (conjunctive) of 'now' formulas, a set (conjunctive) of 'next' formulas and
        a set of acceptance sets, i.e., until or eventually formulas whose eventualities
        are satisfied in that disjunctive term. '''
        # default behavior, keep all terms in now. Overwrite when necessary!
        # determine the DNF
        now = self.in_set_dnf()
        # return set of triples
        return {(conj, frozenset([]), frozenset([])) for conj in now }

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        '''Return the set of all subformulas.'''
        # default behavior, override when necessary
        return set([self])

    def local_alphabet(self)->Set[str]:
        '''Return the set of atomic propositions of the formula. '''
        # default behavior, override when necessary
        return set()

    def filter_symbols(self, symbols: SortedSet, _prop_defs:Dict[str,Set[str]])->SortedSet:
        '''Return symbols that satisfy the propositional formula, using prop_defs
        definition of propositions.'''
        # default all, override if necessary
        return symbols

    def alphabet(self)->SortedSet:
        '''Return the set of propositions in the formula.'''
        return reduce(lambda alpha, phi: alpha.union(phi.local_alphabet()), \
                      self.get_sub_formulas(), SortedSet())

    def _is_liveness_formula(self)->bool:
        '''Return if this is a formula representing a liveness constraint.'''
        # default behavior, update if needed
        return False

    @staticmethod
    def _set_dnf_and(s1: TDisjunctiveNormalForm, s2: TDisjunctiveNormalForm) -> \
        TDisjunctiveNormalForm:
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
    def set_dnf_or(s1: TDisjunctiveNormalForm, s2: TDisjunctiveNormalForm) -> \
         TDisjunctiveNormalForm:
        '''Perform logical or operation on two formulas in set disjunctive normal form.'''
        return s1.union(s2)

    @staticmethod
    def pair_set_dnf_and(
        p1: SortedSet, p2: SortedSet) -> SortedSet:
        '''Perform logical and operation on now, next, acceptance set triples'''
        res = SortedSet()
        for dt1 in p1:
            for dt2 in p2:
                nt_now: TConjunctiveNormalForm = set()
                nt_now.update(dt1[0])
                nt_now.update(dt2[0])
                nt_nxt: TConjunctiveNormalForm = set()
                nt_nxt.update(dt1[1])
                nt_nxt.update(dt2[1])
                nt_acc:Set['LTLSubFormula'] = set()
                nt_acc.update(dt1[2])
                nt_acc.update(dt2[2])
                res.add((frozenset(nt_now), frozenset(nt_nxt), frozenset(nt_acc)))
        return res

    def __eq__(self, other)->bool:
        if not type(self) == type(other): # pylint: disable=unidiomatic-typecheck
            return False
        return self.__eq__same_class__(other)

    def __eq__same_class__(self, other)->bool:
        raise LTLException("to be filled in subclasses to support sorting")

    def __lt__(self, other)->bool:
        if not type(self) == type(other): # pylint: disable=unidiomatic-typecheck
            return type(self).__name__ < type(other).__name__
        return self.__lt__same_class__(other)

    def __lt__same_class__(self, other)->bool:
        raise LTLException("to be filled in subclasses to support sorting")

    def __hash__(self) -> int:
        raise LTLException("to be filled in subclasses")

class LTLFormulaTrue(LTLSubFormula):
    '''Formula true'''

    def __init__(self):
        pass

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        if prop_neg:
            return LTLFormulaFalse()
        return self

    def in_set_dnf(self)->TDisjunctiveNormalForm:
        tt: TConjunctiveNormalForm = frozenset()
        result: TDisjunctiveNormalForm = set()
        result.add(tt)
        return result

    def __str__(self):
        return "true"

    def __eq__same_class__(self, other)->bool:
        return True

    def __lt__same_class__(self, other)->bool:
        return False

    def __hash__(self) -> int:
        return 1

class LTLFormulaFalse(LTLSubFormula):
    '''Formula false.'''

    def __init__(self):
        pass

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        if prop_neg:
            return LTLFormulaTrue()
        else:
            return self

    def in_set_dnf(self)->TDisjunctiveNormalForm:
        return set()

    def filter_symbols(self, symbols: SortedSet, _propDefs:Dict[str,Set[str]])->SortedSet:
        return SortedSet()

    def __str__(self):
        return "false"

    def __eq__same_class__(self, other)->bool:
        return True

    def __lt__same_class__(self, other)->bool:
        return False

    def __hash__(self) -> int:
        return 2


class LTLFormulaProposition(LTLSubFormula):
    '''Proposition formula.'''

    _proposition: str
    _negated: bool

    def __init__(self, p: str, negated:bool = False):
        self._proposition = p
        self._negated = negated

    def proposition(self) -> str:
        '''Get the proposition name'''
        return self._proposition

    def negated(self) -> bool:
        '''Get if the proposition is negated.'''
        return self._negated

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        if prop_neg:
            return LTLFormulaProposition(self._proposition, not self._negated)
        return self

    def local_alphabet(self)->Set[str]:
        return set([self._proposition])

    def filter_symbols(self, symbols: SortedSet, propDefs:Dict[str,Set[str]])->SortedSet:
        if propDefs is None:
            prop_symbols = set([self._proposition])
        else:
            if self._proposition in propDefs:
                prop_symbols = propDefs[self._proposition]
            else:
                prop_symbols = set([self._proposition])

        if self._negated:
            return symbols.difference(prop_symbols)
        return symbols.intersection(prop_symbols)

    def __str__(self):
        if self._negated:
            pre = "not "
        else:
            pre = ""
        if LTL_PROPOSITION_SIMPLE.match(self._proposition):
            return pre + self._proposition
        return pre + "'"+self._proposition.replace("'", "\\'")+"'"

    def __eq__same_class__(self, other)->bool:
        return self._proposition == other.proposition() and self._negated == other.negated()

    def __lt__same_class__(self, other)->bool:
        if self._negated and not other.negated():
            return True
        if other.negated() and not self._negated:
            return False

        return self._proposition < other.proposition()

    def __hash__(self) -> int:
        return hash((self._negated, self._proposition))


class LTLFormulaUntil(LTLSubFormula):
    '''Until formula.'''

    _phi1: LTLSubFormula
    _phi2: LTLSubFormula

    def __init__(self, phi1: LTLSubFormula, phi2: LTLSubFormula):
        self._phi1 = phi1
        self._phi2 = phi2

    def phi1(self):
        '''Return phi1'''
        return self._phi1

    def phi2(self):
        '''Return phi2'''
        return self._phi2

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        if prop_neg:
            # negation converts into Release formula
            return LTLFormulaRelease(self._phi1.in_negation_normal_form_prop(True), \
                                     self._phi2.in_negation_normal_form_prop(True))
        return LTLFormulaUntil(self._phi1.in_negation_normal_form_prop(False), \
                               self._phi2.in_negation_normal_form_prop(False))

    def unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
                                         AbstractSet['LTLSubFormula']]]:
        # unfold in now and next using the following identity
        # phi1 U phi2 = phi2 or (phi1 and X (phi1 U phi2))
        # return a set (disjunction) of triples now, next, acceptance set

        # unfold phi2
        uf2: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]
        uf2 = self._phi2.unfold()  # type: ignore couldn't make the type checker happy

        # unfold phi1
        uf1: SortedSet
        uf1 = self._phi1.unfold()  # type: ignore couldn't make the type checker happy

        # create the next part,  X (phi1 U phi2)
        # note that until formulas create acceptance sets in the equivalent automata
        nu: SortedSet = SortedSet([(frozenset([]), frozenset([self]), \
                                                           frozenset([self]))])

        # return the disjunction (by set union) of uf2  and the conjunction (_pairSetDNFAnd)
        # of uf1 and nu
        return uf2.union(LTLSubFormula.pair_set_dnf_and(uf1, nu))

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        # self, and recursively the subformulas of phi1 and ph2
        return set([self]).union(self._phi1.get_sub_formulas()).union(self._phi2.get_sub_formulas())

    def _is_liveness_formula(self):
        return True

    def __str__(self):
        return "("+str(self._phi1) + ") U (" + str(self._phi2) + ")"

    def __eq__same_class__(self, other)->bool:
        return self._phi1 == other.phi1() and self._phi2 == other.phi2()

    def __lt__same_class__(self, other)->bool:
        if self._phi1.__lt__(other.phi1):
            return True
        if other.phi1().__lt__(self._phi1):
            return False
        return self._phi2 < other.phi2()

    def __hash__(self) -> int:
        return hash((self._phi1, self._phi2, "U"))


class LTLFormulaRelease(LTLSubFormula):
    '''Release formula.'''

    _phi1: LTLSubFormula
    _phi2: LTLSubFormula

    def __init__(self, phi1: LTLSubFormula, phi2: LTLSubFormula):
        self._phi1 = phi1
        self._phi2 = phi2

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        if prop_neg:
            # the negation of a Release formula is an Until formula
            return LTLFormulaUntil(self._phi1.in_negation_normal_form_prop(True), \
                                   self._phi2.in_negation_normal_form_prop(True))
        return LTLFormulaRelease(self._phi1.in_negation_normal_form_prop(False), \
                                 self._phi2.in_negation_normal_form_prop(False))

    def unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
                                        AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # phi1 R phi2 = phi2 and (phi1 or X (phi1 R phi2))
        # phi1 R phi2 = (phi1 and phi2) or (phi2 and X (phi1 R phi2)))
        # return a set (disjunction) of triples now, next, acceptance / eventualities

        uf2: SortedSet
        uf2 = self._phi2.unfold()  # type: ignore couldn't make the type checker happy
        uf1: SortedSet
        uf1 = self._phi1.unfold()  # type: ignore couldn't make the type checker happy
        nr: SortedSet = SortedSet([(frozenset([]), frozenset([self]), \
                                                           frozenset([]))])

        # alternative 1: phi1 and phi2
        alt1 = LTLSubFormula.pair_set_dnf_and(uf1, uf2)
        # alternative 2: phi2 and X (phi1 R phi2)
        alt2 = LTLSubFormula.pair_set_dnf_and(uf2, nr)

        return alt1.union(alt2)

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._phi1.get_sub_formulas()).union(self._phi2.get_sub_formulas())

    def __str__(self):
        return "("+str(self._phi1) + ") R (" + str(self._phi2) + ")"

    def __eq__same_class__(self, other)->bool:
        return self._phi1 == other.phi1() and self._phi2 == other.phi2()

    def __lt__same_class__(self, other)->bool:
        if self._phi1.__lt__(other.phi1):
            return True
        if other.phi1().__lt__(self._phi1):
            return False
        return self._phi2 < other.phi2()

    def __hash__(self) -> int:
        return hash((self._phi1, self._phi2, "R"))


class LTLFormulaImplication(LTLSubFormula):
    '''Implication formula'''

    _phi1: LTLSubFormula
    _phi2: LTLSubFormula

    def __init__(self, phi1: LTLSubFormula, phi2: LTLSubFormula):
        self._phi1 = phi1
        self._phi2 = phi2

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        # use the identity that phi1 => phi2 = not phi1 or phi2
        fs = set()
        fs.add(self._phi1.in_negation_normal_form_prop(not prop_neg))
        fs.add(self._phi2.in_negation_normal_form_prop(prop_neg))
        if prop_neg:
            return LTLFormulaConjunction(fs)
        return LTLFormulaDisjunction(fs)

    def in_set_dnf(self)->TDisjunctiveNormalForm:
        "Not implemented. Implication is rewritten to disjunction before using this function."
        raise LTLException("remove implications first")

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._phi1.get_sub_formulas()).union(self._phi2.get_sub_formulas())

    def __str__(self):
        return "("+str(self._phi1) + ") => (" + str(self._phi1) + ")"

    def __eq__same_class__(self, other)->bool:
        return self._phi1 == other.phi1() and self._phi2 == other.phi2()

    def __lt__same_class__(self, other)->bool:
        if self._phi1.__lt__(other.phi1):
            return True
        if other.phi1().__lt__(self._phi1):
            return False
        return self._phi2 < other.phi2()

    def __hash__(self) -> int:
        return hash((self._phi1, self._phi2, "I"))


class LTLFormulaConjunction(LTLSubFormula):
    '''Conjunction of a set (not necessarily two) of subformulas'''

    _subformulas: Set[LTLSubFormula]

    def __init__(self, subformulas: Set[LTLSubFormula]):
        self._subformulas = subformulas

    def subformulas(self):
        '''Get the direct subformulas.'''
        return self._subformulas

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        fs = {phi.in_negation_normal_form_prop(prop_neg) for phi in self._subformulas}
        if prop_neg:
            return LTLFormulaDisjunction(fs)
        return LTLFormulaConjunction(fs)

    def in_set_dnf(self)->TDisjunctiveNormalForm:
        sub = [phi.in_set_dnf() for phi in self._subformulas]
        result:TDisjunctiveNormalForm = LTLFormulaTrue().in_set_dnf()
        for s in sub:
            result = LTLSubFormula._set_dnf_and(result, s)
        return result

    def unfold(self)->SortedSet:
        # unfold in now and next
        # return a set (disjunction) of pairs now, next

        res:SortedSet = SortedSet([(frozenset([]), frozenset([]), frozenset([]))])

        for phi in self._subformulas:
            unfolded_phi: SortedSet
            unfolded_phi = phi.unfold()  # type: ignore
            res = LTLSubFormula.pair_set_dnf_and(res, unfolded_phi)

        return res

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        l: Callable[[Set[LTLSubFormula], LTLSubFormula], Set[LTLSubFormula]] = lambda res, \
            f: res.union(f.get_sub_formulas())
        return reduce(l, self._subformulas, set())

    def __str__(self):
        return " and ".join(["("+str(phi)+")" for phi in self._subformulas])

    def __eq__same_class__(self, other)->bool:
        if not len(self._subformulas) == len(other.subformulas()):
            return False
        ss = sorted(self._subformulas)
        so = sorted(other.subformulas())
        for i in range(len(self._subformulas)):
            if not ss[i].__eq__(so[i]):
                return False
        return True

    def __lt__same_class__(self, other)->bool:
        if not len(self._subformulas) == len(other.subformulas()):
            return len(self._subformulas) < len(other.subformulas())
        ss = sorted(self._subformulas)
        so = sorted(other.subformulas())
        for i in range(len(self._subformulas)):
            if ss[i].__lt__(so[i]):
                return True
            if so[i].__lt__(ss[i]):
                return False
        return False

    def __hash__(self) -> int:
        ss = sorted(self._subformulas)
        h = hash("C")
        for phi in ss:
            h = hash((h, phi))
        return h


class LTLFormulaDisjunction(LTLSubFormula):
    '''Disjunction formula.'''

    _subformulas: Set[LTLSubFormula]

    def __init__(self, subformulas: Set[LTLSubFormula]):
        self._subformulas = subformulas

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        fs = {phi.in_negation_normal_form_prop(prop_neg) for phi in self._subformulas}
        if prop_neg:
            return LTLFormulaConjunction(fs)
        return LTLFormulaDisjunction(fs)

    def in_set_dnf(self)->TDisjunctiveNormalForm:
        sub = [phi.in_set_dnf() for phi in self._subformulas]
        result = LTLFormulaFalse().in_set_dnf()
        for s in sub:
            result = LTLSubFormula.set_dnf_or(result, s)
        return result

    def unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
                                        AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # return a set (disjunction) of pairs now, next

        res = set()

        for phi in self._subformulas:
            res.update(phi.unfold())

        return res

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:

        l: Callable[[Set[LTLSubFormula], LTLSubFormula], Set[LTLSubFormula]] = \
            lambda res, f: res.union(f.get_sub_formulas())
        return reduce(l, self._subformulas, set())

    def __str__(self):
        return " or ".join(["("+str(phi)+")" for phi in self._subformulas])

    def __eq__same_class__(self, other)->bool:
        if not len(self._subformulas) == len(other.subformulas()):
            return False
        ss = sorted(self._subformulas)
        so = sorted(other.subformulas())
        for i in range(len(self._subformulas)):
            if not ss[i].__eq__(so[i]):
                return False
        return True

    def __lt__same_class__(self, other)->bool:
        if not len(self._subformulas) == len(other.subformulas()):
            return len(self._subformulas) < len(other.subformulas())
        ss = sorted(self._subformulas)
        so = sorted(other.subformulas())
        for i in range(len(self._subformulas)):
            if ss[i].__lt__(so[i]):
                return True
            if so[i].__lt__(ss[i]):
                return False
        return False

    def __hash__(self) -> int:
        ss = sorted(self._subformulas)
        h = hash("D")
        for phi in ss:
            h = hash((h, phi))
        return h


class LTLFormulaNext(LTLSubFormula):
    '''Next formula.'''

    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def subformula(self):
        '''Get the direct subformula.'''
        return self._subformula

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        return LTLFormulaNext(self._subformula.in_negation_normal_form_prop(prop_neg))

    def unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
                                        AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # X phi

        nxt = set([(frozenset([]), frozenset([self._subformula]), frozenset([]))])
        return nxt

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula.get_sub_formulas())

    def __str__(self):
        return "X " + str(self._subformula)

    def __eq__same_class__(self, other)->bool:
        return self._subformula.__eq__(other.subformula())

    def __lt__same_class__(self, other)->bool:
        return self._subformula.__lt__(other.subformula())

    def __hash__(self) -> int:
        return hash((self._subformula,"X"))


class LTLFormulaNegation(LTLSubFormula):
    '''Negation formula.'''

    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def subformula(self):
        '''Get the direct subformula.'''
        return self._subformula

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        return self._subformula.in_negation_normal_form_prop(not prop_neg)

    def in_set_dnf(self)->TDisjunctiveNormalForm:
        # not implemented, assumes the formula is transformed to negation normal form first.
        raise LTLException("Transform to negation normal form first")

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula.get_sub_formulas())

    def __str__(self):
        return "not (" + str(self._subformula) + ")"

    def __eq__same_class__(self, other)->bool:
        return self._subformula.__eq__(other.subformula())

    def __lt__same_class__(self, other)->bool:
        return self._subformula.__lt__(other.subformula())

    def __hash__(self) -> int:
        return hash((self._subformula,"N"))


class LTLFormulaAlways(LTLSubFormula):
    '''Always formula.'''

    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def subformula(self):
        '''Get the direct subformula.'''
        return self._subformula

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        phi = self._subformula.in_negation_normal_form_prop(prop_neg)
        if prop_neg:
            return LTLFormulaEventually(phi)
        return LTLFormulaAlways(phi)

    def unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
                                        AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # G phi = phi and XG phi

        uf:SortedSet
        uf = self._subformula.unfold()  # type: ignore
        ng: SortedSet = SortedSet([(frozenset([]), \
                                    frozenset([self]), frozenset([]))])
        return LTLSubFormula.pair_set_dnf_and(uf, ng)

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula.get_sub_formulas())

    def __str__(self):
        return "G (" + str(self._subformula) + ")"

    def __eq__same_class__(self, other)->bool:
        return self._subformula.__eq__(other.subformula())

    def __lt__same_class__(self, other)->bool:
        return self._subformula.__lt__(other.subformula())

    def __hash__(self) -> int:
        return hash((self._subformula,"G"))

class LTLFormulaEventually(LTLSubFormula):
    '''Eventually formula.'''

    _subformula: LTLSubFormula

    def __init__(self, subformula: LTLSubFormula):
        self._subformula = subformula

    def subformula(self):
        '''Get the direct subformula.'''
        return self._subformula

    def in_negation_normal_form_prop(self, prop_neg: bool)->'LTLSubFormula':
        phi = self._subformula.in_negation_normal_form_prop(prop_neg)
        if prop_neg:
            return LTLFormulaAlways(phi)
        return LTLFormulaEventually(phi)

    def unfold(self)->AbstractSet[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
                                        AbstractSet['LTLSubFormula']]]:
        # unfold in now and next
        # F phi = phi or XF phi

        uf = self._subformula.unfold()
        nf: Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm,AbstractSet['LTLSubFormula']]]
        nf = set([(frozenset([]), frozenset([self]), frozenset([self]))])
        return uf.union(nf)  # type: ignore

    def get_sub_formulas(self)->AbstractSet['LTLSubFormula']:
        return set([self]).union(self._subformula.get_sub_formulas())

    def _is_liveness_formula(self):
        return True

    def __str__(self):
        return "F (" + str(self._subformula) + ")"

    def __eq__same_class__(self, other)->bool:
        return self._subformula.__eq__(other.subformula())

    def __lt__same_class__(self, other)->bool:
        return self._subformula.__lt__(other.subformula())

    def __hash__(self) -> int:
        return hash((self._subformula,"F"))

class LTLFormula:
    '''LTL formula model.'''

    _expression: LTLSubFormula
    _alphabet: Optional[SortedSet]
    _prop_definitions: Dict[str,Set[str]]

    def __init__(self, expression: LTLSubFormula, alphabet: Optional[SortedSet], \
                 definitions: Optional[Dict[str,Set[str]]]):
        self._expression = expression
        self._alphabet = alphabet
        self._prop_definitions = definitions if definitions is not None else {}

    def _set_dnf_and(self, s1: TDisjunctiveNormalForm, s2: TDisjunctiveNormalForm) -> \
          TDisjunctiveNormalForm:
        '''Determine the conjunction of two formulas in set-disjunctive normal form, i.e., a set
        (disjunction) of sets (conjunctions) of formulas.'''
        result = set()
        for dt1 in s1:
            for dt2 in s2:
                nt = set()
                nt.update(dt1)
                nt.update(dt2)
                result.add(frozenset(nt))
        return result

    def _determine_alphabet(self)->SortedSet:
        # if alphabet is explicitly defined return it
        # other wise set of propositions, with those define replace by their sets,

        if self._alphabet is not None:
            return self._alphabet
        # get the propositions from the formula expression
        propositions = self._expression.alphabet()
        # if there are no other definitions, return the propositions from the formula
        if self._prop_definitions is None:
            return propositions

        # determine the alphabet from the defined propositions
        res: SortedSet = SortedSet()
        for p in propositions:
            if p in self._prop_definitions:
                res.update(self._prop_definitions[p])
            else: res.add(p)
        return res

    def as_fsa(self)->Automaton:
        '''Convert the LTL formula to a Büchi automaton that accepts precisely all the
        words that satisfy the formula.'''

        # def _unfold(s: AbstractSet[LTLSubFormula])->SortedSet[Tuple[TConjunctiveNormalForm, \
        #                             TConjunctiveNormalForm,SortedSet['LTLSubFormula']]]:
        def _unfold(s: SortedSet)->SortedSet:
            '''Unfold set of subformulas into DNF and splitting now and next.'''

            # return a set of triples of (frozen) sets of now and next formulas and acceptance sets
            # result:Set[Tuple[TConjunctiveNormalForm,TConjunctiveNormalForm, \
            #                         AbstractSet['LTLSubFormula']]] = set([(frozenset([]), \
            #                             frozenset([]), frozenset([]))])
            result:SortedSet = SortedSet([(frozenset([]), \
                                        frozenset([]), frozenset([]))])
            for phi in s:
                unf: SortedSet
                unf = phi.unfold()  # type: ignore
                result = LTLSubFormula.pair_set_dnf_and(result, unf)
            return result

        def state_string(s: SortedSet)->str:
            return ','.join([str(f) for f in s])

        # def printState(s):
        #     print(state_string(s))

        # def printEdge(s):
        #     print(','.join([str(f) for f in sorted(s)]))

        # set of states by names
        states: Set[str]
        # index of the sets of subformulas corresponding to the states
        state_index: Dict[str,AbstractSet[LTLSubFormula]]
        # a counter of how many states we have created
        state_counter: int
        # names of the initial states
        initial_state_names: Set[str]
        # lookup from set of formulas (as a string) to the name of the corresponding state
        state_index_f: Dict[str,str]
        # keep track of the set of acceptance sets associated with the transitions
        acceptance_sets: Dict[Tuple[str,str,str],Set[str]]

        def add_state(s: SortedSet, initial: bool = False)->bool:
            '''Add a state. Returns True if the state was added, False if it already exists.'''

            nonlocal state_counter
            nonlocal states
            nonlocal state_index
            nonlocal initial_state_names
            nonlocal state_index_f

            # determine string representation for the state
            ss: str = state_string(s)
            # check if it already exists in the state index
            if ss in state_index:
                return False

            # add it to the index
            state_index[ss] = s
            # generate a new name for the state
            state_name = "S"+str(state_counter)
            # update counter
            state_counter+=1
            # add the state
            states.add(state_name)
            if initial:
                initial_state_names.add(state_name)
            state_index_f[ss] = state_name
            return True

        def _acceptance_index(s: str)->int:
            '''Get index of acceptance set from state name'''
            return int(s[s.index("A")+1:s.index(")")])

        # partition transitions on acceptance sets
        # create a state for every incoming combination
        # determine acceptance sets
        # Let FSA reduce Generalized sets

        def add_transition(s: SortedSet, edge_propositional_formula: \
                           AbstractSet[LTLSubFormula], t: SortedSet, \
                            acc: AbstractSet[LTLSubFormula]):
            '''Add a transition from the state corresponding to s, to the state corresponding
            to t, edges labelled by the propositional formula edgeSpec and by acceptance set acc'''

            nonlocal alphabet

            # get the corresponding strings
            ss: str = state_string(s)
            ts: str = state_string(t)

            # determine the symbols of the alphabet that match the propositional formula on the edge
            symbols = reduce(lambda res, f: f.filter_symbols(res, self._prop_definitions), \
                             edge_propositional_formula, alphabet)
            for symbol in sorted(symbols):
                # create an edge from the state corresponding to s to the state
                # corresponding to t, labelled with symbol
                trans = (state_index_f[ss], symbol, state_index_f[ts])
                # create an entry in the acceptanceSets dictionary
                acceptance_sets[trans] = set()
                # add the acceptance sets (as strings) from acc
                acceptance_sets[trans].update({str(a) for a in acc})

        # initialize
        acceptance_sets = SortedDict()
        # initial states from the initial formula expression
        initial_states = self._expression.in_negation_normal_form().in_set_dnf()
        initial_state_names = set()
        # statesToUnfold keeps track of newly created states that need to be unfolded
        # into now and next, initialized with the initial states
        states_to_unfold: SortedSet[SortedSet[LTLSubFormula]] = initial_states.copy()  # type: ignore

        # check if we have an explicit alphabet, otherwise, compute
        alphabet = self._determine_alphabet()

        # Create the automaton
        state_counter = 1
        state_index_f = {}
        states = set()
        state_index = {}

        # for s in sorted(states_to_unfold):
        #     for t in sorted(s):
        #         print(t)

        # add the states we start from as initial states
        for s in sorted(states_to_unfold):
            add_state(s, True)

        # as long as we states that still need to be unfolded
        while len(states_to_unfold) > 0:
            # take one state from the set and remove it
            s = next(iter(states_to_unfold))
            states_to_unfold.remove(s)

            # determine outgoing transitions from unfolded state
            transitions = _unfold(s)
            for t in sorted(transitions):
                t1: SortedSet[LTLSubFormula] = t[1]  # type: ignore
                # add state if it doesn't exist yet
                if add_state(t1):
                    # the state is new, so needs unfolding
                    states_to_unfold.add(t1)
                # add a transition from s to the new state with appropriate labels
                # of propositional formula and acceptance sets
                add_transition(s, t[0], t[1], t[2])

        # for all states collect the incoming acceptance labels
        # and store them in a dictionary
        # create initial dictionary with empty sets
        # maps a state (str) to a set of set of str
        acceptance_sets_states: Dict[str,Set[AbstractSet[str]]]
        acceptance_sets_states = {s:set() for s in states}
        # for all transitions, keys of acceptanceSets
        for t in acceptance_sets:
            # add the acceptance set to the set of acceptance sets of the target state
            acceptance_sets_states[t[2]].add(frozenset(acceptance_sets[t]))

        # collect all the acceptance labels from all states
        acceptance_labels: Set[AbstractSet[str]]
        acceptance_labels = reduce(lambda res, acc: res.union(acc), \
                                   acceptance_sets_states.values(), set())
        # print(acceptanceLabels)

        # associate all (non empty) acceptance labels with a unique number starting from 1
        state_labels_acceptance: Dict[AbstractSet[str],int] = {}
        k = 1
        for a in sorted(acceptance_labels):
            if len(a) > 0:
                state_labels_acceptance[a] = k
                k += 1

        # ultimate state names are formed as a combination of state name and acceptance label number
        def build_state_name(s, n):
            return f"({s},A{str(n)})"

        f = Automaton()

        # add states without acceptance label
        for s in sorted(states):
            f.add_state(build_state_name(s, 0))

        # make initial states
        for s in sorted(initial_state_names):
            f.make_initial_state(build_state_name(s, 0))

        # for all states with incoming acceptance labels make different states
        for s in sorted(acceptance_sets_states):
            for a in sorted(acceptance_sets_states[s]):
                if len(a) > 0:
                    f.add_state(build_state_name(s, state_labels_acceptance[a]))

        # construct appropriate transitions
        for t in acceptance_sets:
            a = frozenset(acceptance_sets[t])
            if len(a) == 0:
                target_state = build_state_name(t[2], 0)
            else:
                target_state = build_state_name(t[2], state_labels_acceptance[a])
            for s in sorted(f.states()):
                if s.startswith("("+t[0]):
                    f.add_transition(s, t[1], target_state)

        # compute acceptance sets
        # one for each non-empty set
        acceptance_sets_keys = reduce(lambda res, acc: res.union(acc), acceptance_labels, set())
        generalized_acceptance_sets = []
        for k in sorted(acceptance_sets_keys):
            acceptance_state_labels = [state_labels_acceptance[l] for l in acceptance_labels \
                                       if k in l]
            acc_set_of_states = [s for s in f.states() if _acceptance_index(s) in \
                                  acceptance_state_labels]
            generalized_acceptance_sets.append(frozenset( \
                set(f.states()).difference(acc_set_of_states)))

        # if there are no acceptance sets, we need to make all states accepting
        if len(generalized_acceptance_sets) == 0:
            for s in sorted(f.states()):
                f.make_final_state(s)
        else:
            # set the first acceptance set as accepting states
            for s in sorted(generalized_acceptance_sets[0]):
                f.make_final_state(s)
            # add all following ones as transformations
            f = f.add_generalized_buchi_acceptance_sets(generalized_acceptance_sets[1:])

        return f


    def as_dsl(self, name):
        '''Convert formula to DSL representation.'''
        return f'ltl formula {name} = {str(self)}'

    @staticmethod
    def from_dsl(ltl_string)->Tuple[str,'LTLFormula']:
        '''Parse formula from DSL string.'''
        factory = {}
        factory['Proposition'] = LTLFormulaProposition
        factory['Until'] = LTLFormulaUntil
        factory['Release'] = LTLFormulaRelease
        factory['Next'] = LTLFormulaNext
        factory['Always'] = LTLFormulaAlways
        factory['Negation'] = LTLFormulaNegation
        factory['Eventually'] = LTLFormulaEventually
        factory['Disjunction'] = LTLFormulaDisjunction
        factory['Conjunction'] = LTLFormulaConjunction
        factory['True'] = LTLFormulaTrue
        factory['False'] = LTLFormulaFalse
        (name, form) = parse_ltl_dsl(ltl_string, factory)
        if form is None or name is None:
            sys.exit(1)
        phi: LTLSubFormula
        alphabet: Optional[SortedSet]
        definitions: Optional[Dict[str,Set[str]]]
        (phi, alphabet, definitions) = form
        return name, LTLFormula(phi, alphabet, definitions)

    def __str__(self)->str:
        return str(self._expression)
