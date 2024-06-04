'''Library for regular expression analysis.'''
from enum import Enum
import re
import sys
from typing import Dict, List, Optional, Tuple,Set

from finitestateautomata.libfsa import Automaton
from finitestateautomata.libregexgrammar import parse_reg_ex_dsl
from finitestateautomata.utils.utils import FiniteStateAutomataException

RegExTermType = Enum('RegExTermType', 'none, emptyset emptyword concat alternatives kleene')

REGEX_EMPTY_SET =  re.compile(r"^\\o", re.MULTILINE|re.DOTALL)
REGEX_EMPTY_WORD = re.compile(r"^\\e", re.MULTILINE|re.DOTALL)
REGEX_CONCATENATION = re.compile(r"^\.\(", re.MULTILINE|re.DOTALL)
REGEX_ALTERNATIVE = re.compile(r"^\+\(", re.MULTILINE|re.DOTALL)
REGEX_KLEENE = re.compile(r"^\*\(", re.MULTILINE|re.DOTALL)
REGEX_LETTER = re.compile(r"^([a-zA-Z]|'[^']+\')", re.MULTILINE|re.DOTALL)
REGEX_LETTER_SIMPLE = re.compile(r"^[a-zA-Z]$")

class RegExException(FiniteStateAutomataException):
    '''Regular Exception related exception'''

class RegExTerm:
    '''Regular Expression'''

    def __init__(self):
        pass

    # parse expression. return expression and remainder of the string
    @staticmethod
    def from_string(s: str)->Tuple['RegExTerm',str]:
        '''Parse expression. return expression and remainder of the string.'''
        if REGEX_EMPTY_SET.match(s):
            return (RegExTermEmptySet(), s[2:])
        if REGEX_EMPTY_WORD.match(s):
            return (RegExTermEmptyWord(), s[2:])
        if REGEX_CONCATENATION.match(s):
            return RegExTermConcatenation.from_string(s)
        if REGEX_ALTERNATIVE.match(s):
            return RegExTermAlternatives.from_string(s)
        if REGEX_KLEENE.match(s):
            return RegExTermKleene.from_string(s)
        if REGEX_LETTER.match(s):
            return RegExTermLetter.from_string(s)
        raise RegExException('malformed regular expression')

    def sub_expressions(self)->Set['RegExTerm']:
        '''Return the set of all subexpressions.'''
        return set()

    def collect_alphabet(self, result: Set[str])->None:
        '''Collet the alphabet of the regex term into the set result.'''
        for e in self.sub_expressions():
            e.collect_alphabet(result)

    def is_finite_reg_ex(self)->bool:
        '''Return if the expression represents a finite regular expression, not an
        omega-regular expression.'''
        # default result, override if necessary
        return True

    def is_omega_reg_ex(self)->bool:
        '''Return if the expression represent an omega-regular regular expression,
        not a finite expression.'''
        # default result, override if necessary
        return True

    def binding_level(self)->int:
        '''Return binding level to determine how to place parentheses.'''
        raise RegExException("Should have been filled in subclasses to determine parentheses")

    def _parenthesize(self, e: 'RegExTerm')->str:
        '''Convert to string with the right parentheses.'''
        if e.binding_level() < self.binding_level():
            return '('+str(e)+')'
        return str(e)

    def __str__(self)->str:
        raise RegExException("to be filled in subclasses")

    def simplify(self)->'RegExTerm':
        '''Return a simplified expression.'''
        return self

    def is_empty_word(self)->bool:
        '''Return whether the expression represents the empty word.'''
        return False

    def is_empty_set(self)->bool:
        '''Return whether the expression represents the empty language.'''
        return False

    def as_nba(self)->Automaton:
        '''Converts omega-regular expression to an equivalent NBA.'''
        if not self.is_omega_reg_ex():
            raise RegExException('Not an omega-regular expression.')
        return self.exp_as_nba()

    def as_fsa(self)->Automaton:
        '''Converts finite regular expression to an equivalent FSA.'''
        if not self.is_finite_reg_ex():
            raise RegExException('Not a finite regular expression.')
        return self.exp_as_fsa()

    def exp_as_nba(self)->Automaton:
        '''Perform conversion to an NBA.'''
        raise RegExException("Overwrite in subclasses")

    def exp_as_fsa(self)->Automaton:
        '''Perform conversion to an FSA.'''
        raise RegExException("Overwrite in subclasses")

    def __eq__(self, other)->bool:
        if not type(self) == type(other): # pylint: disable=unidiomatic-typecheck
            return False
        return self.__eq__same_class__(other)

    def __eq__same_class__(self, other)->bool:
        raise RegExException("to be filled in subclasses to support sorting")

    def __lt__(self, other)->bool:
        if not type(self) == type(other): # pylint: disable=unidiomatic-typecheck
            return type(self).__name__ < type(other).__name__
        return self.__lt__same_class__(other)

    def __lt__same_class__(self, other)->bool:
        raise RegExException("to be filled in subclasses to support sorting")

    def __hash__(self) -> int:
        raise RegExException("to be filled in subclasses")


    @staticmethod
    def from_fsa(a: Automaton)->'RegExTerm':
        '''Convert FSA to regex.'''

        # map a vertex to a tuple
        # - set of vertices with backward transition,
        # - set of vertices with forward transition,
        # - a dict with for every forward next vertex, a set of regular
        #   expressions with which the transition is labelled
        vertices_map: Dict[str,Tuple[Set[str],Set[str],Dict[str,Set['RegExTerm']]]]

        def _add_transition(u: str, v: str, reg_exp: 'RegExTerm'):
            '''Add (to) transition u->v if it exists, or create transition.'''
            nonlocal vertices_map
            if not v in vertices_map[u][1]:
                # add the forward link
                vertices_map[u][1].add(v)
                # add the backward link
                vertices_map[v][0].add(u)
                # create a set for labels
                vertices_map[u][2][v] = set()
            # add the regex to the transition labels
            vertices_map[u][2][v].add(reg_exp)

        def _remove_transition(u,v):
            '''remove transition u->v'''
            nonlocal vertices_map
            # Remove the forward link
            vertices_map[u][1].remove(v)
            # remove the regex labels
            del vertices_map[u][2][v]
            # remove the backward link
            vertices_map[v][0].remove(u)

        def _remove_cycle(s)->Optional['RegExTerm']:
            '''If s has a cycle, return a regular expression that represents the
            alternatives of the labels on the cycle. Returns None if there is no cycle on s.'''
            nonlocal vertices_map
            # if s has transition to itself
            if s in vertices_map[s][1]:
                # create an alternatives term with each of the labels on the cycle
                reg_exp = RegExTermAlternatives(list(vertices_map[s][2][s]))
                # remove the cycle forward and backward links and labels
                vertices_map[s][0].remove(s)
                vertices_map[s][1].remove(s)
                del vertices_map[s][2][s]
                # return the alternatives expression created from the cycle
                return reg_exp
            # there is no cycle
            return None

        # create a graph
        states: List[str] = a.states_in_bfs_order()
        vertices_map = {}

        # add all states
        for s in a.states():
            # incoming states, outgoing states, map of outgoing states to sets of alternative REs
            vertices_map[s] = (set(), set(), {})

        # add all edges
        for t in a.transitions():
            _add_transition(t[0], t[2], RegExTermLetter(t[1]))
        for t in a.epsilon_transitions():
            _add_transition(t[0], t[1], RegExTermEmptyWord())

        # make a single source connected to all initial states
        if '_src' in states:
            raise RegExException("Automaton has a clashing state name: '_src'")
        vertices_map['_src'] = (set(), set(), {})
        for s in a.initial_states():
            _add_transition('_src', s, RegExTermEmptyWord())

        # make a single sink, connected to all final states
        if '_snk' in states:
            raise RegExException("Automaton has a clashing state name: '_snk'")
        vertices_map['_snk'] = (set(), set(), {})
        for s in a.final_states():
            _add_transition(s, '_snk', RegExTermEmptyWord())

        # eliminate all nodes for all states of the FSA by bypassing pairs of
        # incoming and outgoing edges
        for s in states:
            # eliminate state s
            # incorporating self-loops
            cycle_reg_ex = _remove_cycle(s)
            # for all pairs of incoming and outgoing edges make shortcut
            s_in = vertices_map[s][0].copy()
            s_out = vertices_map[s][1].copy()
            for u in s_in:
                for v in s_out:
                    re1 = RegExTermAlternatives(list(vertices_map[u][2][s]))
                    re2 = RegExTermAlternatives(list(vertices_map[s][2][v]))
                    if cycle_reg_ex:
                        re_new = RegExTermConcatenation([re1, RegExTermKleene(cycle_reg_ex), re2])
                    else:
                        re_new = RegExTermConcatenation([re1, re2])
                    # add (to) transition u->v
                    _add_transition(u, v, re_new)
            for u in s_in:
                # remove transition u->s
                _remove_transition(u, s)
            for v in s_out:
                # remove transition s->v
                _remove_transition(s, v)

        # check if there is a path from src to snk
        if not '_snk' in vertices_map['_src'][1]:
            return RegExTermEmptySet()
        return RegExTermAlternatives(list(vertices_map['_src'][2]['_snk']))

class RegExTermEmptySet(RegExTerm):
    '''Regular expression for the empty language.'''

    def __init__(self):
        pass

    def binding_level(self)->int:
        return 3

    def __str__(self)->str:
        return "\\o"

    def is_empty_set(self)->bool:
        return True

    def exp_as_fsa(self)->Automaton:
        result = Automaton()
        si = result.add_state_unique("S")
        result.make_initial_state(si)
        return result

    def exp_as_nba(self)->Automaton:
        return self.exp_as_fsa()

    def __eq__same_class__(self, other)->bool:
        return True

    def __lt__same_class__(self, other)->bool:
        return False

    def __hash__(self) -> int:
        return 1


class RegExTermEmptyWord(RegExTerm):
    '''Regular expression for the empty word.'''

    def __init__(self):
        pass

    def binding_level(self)->int:
        return 3

    def is_omega_reg_ex(self)->bool:
        return False

    def __str__(self)->str:
        return "\\e"

    def is_empty_word(self)->bool:
        return True

    def exp_as_fsa(self)->Automaton:
        result = Automaton()
        si = result.add_state_unique("S")
        result.make_initial_state(si)
        result.make_final_state(si)
        return result

    def __eq__same_class__(self, other)->bool:
        return True

    def __lt__same_class__(self, other)->bool:
        return False

    def __hash__(self) -> int:
        return 2

class RegExTermConcatenation(RegExTerm):
    '''Regular expression for concatenation.'''

    _expressions: List[RegExTerm]

    def __init__(self, expressions: List[RegExTerm]):
        self._expressions = expressions

    def sub_expressions(self)->List[RegExTerm]:
        return self._expressions

    def is_finite_reg_ex(self)->bool:
        for k in range(len(self._expressions)-1):
            if not self._expressions[k].is_finite_reg_ex():
                raise RegExException('Invalid regular expression.')
        if len(self._expressions) == 0:
            return True
        return self._expressions[-1].is_finite_reg_ex()

    def is_omega_reg_ex(self)->bool:
        for k in range(len(self._expressions)-1):
            if not self._expressions[k].is_finite_reg_ex():
                raise RegExException('Invalid regular expression.')
        if len(self._expressions) == 0:
            return True
        return self._expressions[-1].is_omega_reg_ex()

    def simplify(self)->RegExTerm:

        simplified_expressions = [e.simplify() for e in self._expressions]
        if RegExTermEmptySet() in simplified_expressions:
            return RegExTermEmptySet()
        non_trivial_expr = list(filter(lambda e: not e.is_empty_word(), simplified_expressions))
        if len(non_trivial_expr) == 0:
            return RegExTermEmptyWord()
        if len(non_trivial_expr) == 1:
            return next(iter(non_trivial_expr))
        return RegExTermConcatenation(non_trivial_expr)

    def __eq__same_class__(self, other)->bool:
        if not len(self._expressions) == len(other.sub_expressions()):
            return False
        for i, e in enumerate(self._expressions):
            if not e.__eq__(other.expressions[i]):
                return False
        return True

    def __lt__same_class__(self, other)->bool:
        if not len(self._expressions) == len(other.sub_expressions()):
            return len(self._expressions) < len(other.sub_expressions())
        for i, e in enumerate(self._expressions):
            if e.__lt__(other.expressions[i]):
                return True
        return False

    def __hash__(self) -> int:
        return hash((3, hash(tuple(self._expressions))))

    @staticmethod
    def from_string(s: str)->Tuple[RegExTerm,str]:
        # read subexpressions until the terminating ')' symbol
        s = s[2:]
        expressions = []
        while not re.match(r"\)", s):
            (exp, s) = RegExTerm.from_string(s)
            expressions.append(exp)
        return (RegExTermConcatenation(expressions), s[1:])

    def binding_level(self)->int:
        return 1

    def __str__(self)->str:
        return ".".join([self._parenthesize(e) for e in self._expressions])

    def exp_as_fsa(self)->Automaton:
        # get automata for subexpressions
        expr_fsa = [e.exp_as_fsa() for e in self._expressions]
        ia = expr_fsa[0]
        fa = expr_fsa[-1]

        # build a new automaton
        result = Automaton()
        state_map = {}
        # add all states
        for a in expr_fsa:
            state_map[a] = {}
            for s in sorted(a.states()):
                ns = result.add_state_unique(s)
                state_map[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in sorted(a.transitions()):
                result.add_transition(state_map[a][src], symbol, state_map[a][dst])
            # add all epsilon transitions
            for (src, dst) in sorted(a.epsilon_transitions()):
                result.add_epsilon_transition(state_map[a][src], state_map[a][dst])
        # add initial states
        for s in sorted(ia.initial_states()):
            result.make_initial_state(state_map[ia][s])
        # add final states
        for s in sorted(fa.final_states()):
            result.make_final_state(state_map[fa][s])
        # connect automata, n to n+1 while n+1 < len exprFSA
        n = 0
        while n+1 < len(expr_fsa):
            aa = expr_fsa[n]
            ab = expr_fsa[n+1]
            for s in sorted(aa.final_states()):
                for t in sorted(ab.initial_states()):
                    result.add_epsilon_transition(state_map[aa][s], state_map[ab][t])
            n += 1

        return result

    def exp_as_nba(self)->Automaton:
        # get automata for subexpressions
        expr_fsa = [e.exp_as_fsa() for e in self._expressions[:-1]]
        if len(expr_fsa) > 0:
            expr_nba = self._expressions[-1].exp_as_nba()
            expr_fsa.append(expr_nba)

        ia = expr_fsa[0]
        fa = expr_fsa[-1]

        # build a new automaton
        result = Automaton()
        state_map = {}
        # add all states
        for a in expr_fsa:
            state_map[a] = {}
            for s in sorted(a.states()):
                ns = result.add_state_unique(s)
                state_map[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in sorted(a.transitions()):
                result.add_transition(state_map[a][src], symbol, state_map[a][dst])
            # add all epsilon transitions
            for (src, dst) in sorted(a.epsilon_transitions()):
                result.add_epsilon_transition(state_map[a][src], state_map[a][dst])
        # add initial states
        for s in sorted(ia.initial_states()):
            result.make_initial_state(state_map[ia][s])
        # add final states
        for s in sorted(fa.final_states()):
            result.make_final_state(state_map[fa][s])
        # connect automata, n to n+1 while n+1 < len exprFSA
        n = 0
        while n+1 < len(expr_fsa):
            aa = expr_fsa[n]
            ab = expr_fsa[n+1]
            for s in sorted(aa.final_states()):
                for t in sorted(ab.initial_states()):
                    result.add_epsilon_transition(state_map[aa][s], state_map[ab][t])
            n += 1

        return result

class RegExTermAlternatives(RegExTerm):
    '''Regular expression alternatives.'''

    _expressions: List[RegExTerm]

    def __init__(self, expressions: List[RegExTerm]):
        self._expressions = expressions

    def sub_expressions(self)->List[RegExTerm]:
        return self._expressions

    def is_finite_reg_ex(self)->bool:
        for e in self._expressions:
            if not e.is_finite_reg_ex():
                return False
        return True

    def is_omega_reg_ex(self)->bool:
        for e in self._expressions:
            if not e.is_omega_reg_ex():
                return False
        return True

    def simplify(self)->RegExTerm:
        simplified_expressions = [e.simplify() for e in self._expressions]
        non_trivial_expr = list(filter(lambda e: not e.is_empty_set(), simplified_expressions))
        if len(non_trivial_expr) == 0:
            return RegExTermEmptySet()
        unique_expressions = set(non_trivial_expr)
        if len(unique_expressions) == 1:
            return next(iter(unique_expressions))
        return RegExTermAlternatives(list(unique_expressions))

    @staticmethod
    def from_string(s: str)->Tuple[RegExTerm,str]:
        # read subexpressions until the terminating ')' symbol
        s = s[2:]
        expressions = []
        while not re.match(r"\)", s):
            (exp, s) = RegExTerm.from_string(s)
            expressions.append(exp)
        return (RegExTermAlternatives(expressions), s[1:])

    def binding_level(self)->int:
        return 0

    def __str__(self)->str:
        return "+".join([self._parenthesize(e) for e in self._expressions])

    def exp_as_fsa(self)->Automaton:
        # get automata for subexpressions

        expr_fsa = [e.exp_as_fsa() for e in sorted(self._expressions)]

        # build a new automaton
        result = Automaton()
        state_map = {}
        # add all states
        for a in expr_fsa:
            state_map[a] = {}
            for s in sorted(a.states()):
                ns = result.add_state_unique(s)
                state_map[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in sorted(a.transitions()):
                result.add_transition(state_map[a][src], symbol, state_map[a][dst])
            # add all epsilon transitions
            for (src, dst) in sorted(a.epsilon_transitions()):
                result.add_epsilon_transition(state_map[a][src], state_map[a][dst])
            # add initial states
            for s in sorted(a.initial_states()):
                result.make_initial_state(state_map[a][s])
            # add final states
            for s in sorted(a.final_states()):
                result.make_final_state(state_map[a][s])

        return result

    def exp_as_nba(self)->Automaton:
        # get automata for subexpressions
        expr_nba = [e.as_nba() for e in sorted(self._expressions)]

        # build a new automaton
        result = Automaton()
        state_map = {}
        # add all states
        for a in expr_nba:
            state_map[a] = {}
            for s in sorted(a.states()):
                ns = result.add_state_unique(s)
                state_map[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in sorted(a.transitions()):
                result.add_transition(state_map[a][src], symbol, state_map[a][dst])
            # add all epsilon transitions
            for (src, dst) in sorted(a.epsilon_transitions()):
                result.add_epsilon_transition(state_map[a][src], state_map[a][dst])
            # add initial states
            for s in sorted(a.initial_states()):
                result.make_initial_state(state_map[a][s])
            # add final states
            for s in sorted(a.final_states()):
                result.make_final_state(state_map[a][s])

        return result

    def __eq__same_class__(self, other)->bool:
        if not len(self._expressions) == len(other.sub_expressions()):
            return False
        ss = sorted(self._expressions)
        so = sorted(other.sub_expressions())
        for i, e in enumerate(ss):
            if not e.__eq__(so[i]):
                return False
        return True

    def __lt__same_class__(self, other)->bool:
        if not len(self._expressions) == len(other.sub_expressions()):
            return len(self._expressions) < len(other.sub_expressions())
        ss = sorted(self._expressions)
        so = sorted(other.sub_expressions())
        for i, e in enumerate(ss):
            if e.__lt__(so[i]):
                return True
        return False

    def __hash__(self) -> int:
        return hash((4, hash(tuple(self._expressions))))


class RegExTermKleene(RegExTerm):
    '''Regular expression Kleene star.'''

    _expression: RegExTerm

    def __init__(self, expression: RegExTerm):
        self._expression = expression

    def subexpression(self):
        '''Get subexpression'''
        return self._expression

    def is_finite_reg_ex(self)->bool:
        if not self._expression.is_finite_reg_ex():
            raise RegExException('Illegal regular expression.')
        return True

    def is_omega_reg_ex(self)->bool:
        return False

    def simplify(self)->RegExTerm:
        simplified_expression = self._expression.simplify()
        if simplified_expression.is_empty_set():
            return RegExTermEmptySet()
        if simplified_expression.is_empty_word():
            return RegExTermEmptyWord()
        return RegExTermKleene(simplified_expression)

    def sub_expressions(self)->Set[RegExTerm]:
        res = set()
        res.add(self._expression)
        return res

    @staticmethod
    def from_string(s: str)->Tuple[RegExTerm,str]:
        # read subexpression and the terminating ')' symbol
        s = s[2:]
        (exp, s) = RegExTerm.from_string(s)
        return (RegExTermKleene(exp), s[1:])

    def binding_level(self)->int:
        return 2

    def __str__(self)->str:
        return self._parenthesize(self._expression)+"*"

    def exp_as_fsa(self)->Automaton:
        # get automaton for subexpression
        result = self._expression.exp_as_fsa()

        # add a new state that is initial and final
        sif = result.add_state_unique("S")

        # add feedback transitions
        for s in sorted(result.initial_states()):
            result.add_epsilon_transition(sif, s)
        for s in sorted(result.final_states()):
            result.add_epsilon_transition(s, sif)

        # make the new state the only initial state
        result.clear_initial_states()
        result.make_initial_state(sif)
        result.clear_final_states()
        result.make_final_state(sif)

        return result

    def __eq__same_class__(self, other)->bool:
        return self._expression.__eq__(other.subexpression())

    def __lt__same_class__(self, other)->bool:
        return self._expression.__lt__(other.subexpression())

    def __hash__(self) -> int:
        return hash((5, self._expression))


class RegExTermOmega(RegExTerm):
    '''Regular expression omega repetition.'''

    _expression: RegExTerm

    def __init__(self, expression: RegExTerm):
        self._expression = expression

    def subexpression(self):
        '''Get subexpression'''
        return self._expression

    def is_finite_reg_ex(self)->bool:
        return False

    def is_omega_reg_ex(self)->bool:
        if not self._expression.is_finite_reg_ex():
            raise RegExException('Illegal regular expression.')
        return True

    def simplify(self)->RegExTerm:
        simplified_expression = self._expression.simplify()
        if simplified_expression.is_empty_set():
            return RegExTermEmptySet()
        if simplified_expression.is_empty_word():
            raise RegExException('Invalid omega-regular expression')
        return RegExTermOmega(simplified_expression)

    def sub_expressions(self)->Set[RegExTerm]:
        res = set()
        res.add(self._expression)
        return res

    @staticmethod
    def from_string(s: str)->Tuple[RegExTerm,str]:
        # read subexpression and the terminating ')' symbol
        s = s[2:]
        (exp, s) = RegExTerm.from_string(s)
        return (RegExTermOmega(exp), s[1:])

    def binding_level(self)->int:
        return 2

    def __str__(self)->str:
        return self._parenthesize(self._expression)+"**"

    def exp_as_nba(self)->Automaton:
        # get automaton for subexpression
        result = self._expression.exp_as_fsa()

        # add feedback transitions
        for s in sorted(result.initial_states()):
            for t in sorted(result.final_states()):
                result.add_epsilon_transition(t, s)

        return result

    def __eq__same_class__(self, other)->bool:
        return self._expression.__eq__(other.subexpression())

    def __lt__same_class__(self, other)->bool:
        return self._expression.__lt__(other.subexpression())

    def __hash__(self) -> int:
        return hash((6, self._expression))


class RegExTermLetter(RegExTerm):
    '''Regular expression letter symbol.'''

    _letter: str


    def __init__(self, letter: str):
        self._letter = letter

    def letter(self):
        '''Get letter.'''
        return self._letter

    def is_omega_reg_ex(self)->bool:
        return False

    def collect_alphabet(self, result: Set[str])->None:
        result.add(self._letter)

    @staticmethod
    def from_string(s: str)->Tuple[RegExTerm,str]:
        m = REGEX_LETTER.search(s)
        if m is None:
            raise RegExException("Failed to match REGEX_LETTER expression.")
        letter = m.group(0)
        return (RegExTermLetter(letter), s[len(letter):])

    def binding_level(self)->int:
        return 3

    def __str__(self)->str:
        if REGEX_LETTER_SIMPLE.match(self._letter):
            return self._letter
        return "'"+self._letter.replace("'", "\\'")+"'"

    def exp_as_fsa(self)->Automaton:

        result = Automaton()
        si = result.add_state_unique("S")
        sf = result.add_state_unique("S")
        symbol = self._letter.replace("'", "")
        result.add_transition(si, symbol, sf)
        result.make_initial_state(si)
        result.make_final_state(sf)
        return result

    def __eq__same_class__(self, other)->bool:
        return self._letter == other.letter()

    def __lt__same_class__(self, other)->bool:
        return self._letter < other.letter()

    def __hash__(self) -> int:
        return hash((7, self._letter))

class RegEx:
    '''Regular Expression model.'''

    _expression: RegExTerm
    _name: str

    def __init__(self, name: str, expression: RegExTerm):
        self._expression = expression
        self._name = name

    def as_fsa(self)->Automaton:
        '''Convert regular expression to FSA.'''
        if not self._expression.is_finite_reg_ex():
            raise RegExException('Not a finite regular expression.')
        return self._expression.as_fsa()

    def as_nba(self)->Automaton:
        '''Convert Omega regular expression to NBA.'''
        if not self._expression.is_omega_reg_ex():
            raise RegExException('Not an omega regular expression.')
        return self._expression.as_nba()

    def alphabet(self)->Set[str]:
        '''Return the alphabet of the regex '''
        result: Set[str] = set()
        self._expression.collect_alphabet(result)
        return result

    def is_omega_reg_ex(self)->bool:
        '''Test if the regular expression is an omega regular expression.'''
        return self._expression.is_omega_reg_ex()

    def is_finite_reg_ex(self)->bool:
        '''Test if the regular expression is a finite regular expression.'''
        return self._expression.is_finite_reg_ex()

    def as_dsl(self, name: str):
        '''Convert regular expression to DSL.'''
        return f'regular expression {name} = {str(self)}'

    @staticmethod
    def from_dsl(regex_string)->Tuple[str,'RegEx']:
        '''Create regular expression model from DSL.'''
        factory = {}
        factory['Letter'] = RegExTermLetter
        factory['Kleene'] = RegExTermKleene
        factory['Omega'] = RegExTermOmega
        factory['Alternatives'] = RegExTermAlternatives
        factory['Concatenations'] = RegExTermConcatenation
        factory['EmptyLanguage'] = RegExTermEmptySet
        factory['EmptyWord'] = RegExTermEmptyWord
        (name, expression) = parse_reg_ex_dsl(regex_string, factory)
        if name is None or expression is None:
            sys.exit(1)
        return name, RegEx(name, expression)

    @staticmethod
    def from_string(regex_string):
        '''Create regular expression from string.'''

        # find the name
        reg_ex_regex = r".*regular\s+expression\s+(?P<name>[^\s]+)\s*=\s*(?P<regex>[^\s]*?)\s*$"
        match = re.search(reg_ex_regex, regex_string, re.MULTILINE|re.DOTALL)
        if match is None:
            raise RegExException("Input is not a valid RegEx")
        name = match.group('name')

        (regex, _) = RegExTerm.from_string(match.group('regex'))

        return (name, RegEx(name, regex))

    @staticmethod
    def from_fsa(a, name):
        '''Convert FSA to regular expression.'''
        return RegEx(name, RegExTerm.from_fsa(a).simplify())

    def __str__(self)->str:
        return str(self._expression)
