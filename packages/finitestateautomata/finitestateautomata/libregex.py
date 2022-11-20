from enum import Enum
import re
from typing import Dict, List, Optional, Tuple,Set

from finitestateautomata.libfsa import Automaton
from finitestateautomata.libregexgrammar import parseRegExDSL

RegExTermType = Enum('RegExTermType', 'none, emptyset emptyword concat alternatives kleene')

REGEXEMPTYSET =  re.compile(r"^\\o", re.MULTILINE|re.DOTALL)
REGEXEMPTYWORD = re.compile(r"^\\e", re.MULTILINE|re.DOTALL)
REGEXCONCATENATION = re.compile(r"^\.\(", re.MULTILINE|re.DOTALL)
REGEXALTERNATIVE = re.compile(r"^\+\(", re.MULTILINE|re.DOTALL)
REGEXKLEENE = re.compile(r"^\*\(", re.MULTILINE|re.DOTALL)
REGEXLETTER = re.compile(r"^([a-zA-Z]|'[^']+\')", re.MULTILINE|re.DOTALL)
REGEXLETTERSIMPLE = re.compile(r"^[a-zA-Z]$")

class RegExException(Exception):
    pass

class RegExTerm(object):

    def __init__(self):
        pass

    # parse expression. return expression and remainder of the string
    @staticmethod
    def fromString(s: str)->Tuple['RegExTerm',str]:
        '''Parse expression. return expression and remainder of the string.'''
        if REGEXEMPTYSET.match(s):
            return (RegExTermEmptySet(), s[2:])
        if REGEXEMPTYWORD.match(s):
            return (RegExTermEmptyWord(), s[2:])
        if REGEXCONCATENATION.match(s):
            return RegExTermConcatenation.fromString(s)
        if REGEXALTERNATIVE.match(s):
            return RegExTermAlternatives.fromString(s)
        if REGEXKLEENE.match(s):
            return RegExTermKleene.fromString(s)
        if REGEXLETTER.match(s):
            return RegExTermLetter.fromString(s)
        raise Exception('malformed regular expression')

    def subExpressions(self)->Set['RegExTerm']:
        '''Return the set of all subexpressions.'''
        return set()

    def _collectAlphabet(self, result: Set[str])->None:
        '''Collet the alphabet of the regex term into the set result.'''
        for e in self.subExpressions():
            e._collectAlphabet(result)

    def isFiniteRegEx(self)->bool:
        '''Return if the expression represents a finite regular expression, not an omega-regular expression.'''
        # default result, override if necessary
        return True

    def isOmegaRegEx(self)->bool:
        '''Return if the expression represent an omega-regular regular expression, not a finite expression.'''
        # default result, override if necessary
        return True

    def _bindingLevel(self)->int:
        '''Return binding level to determine how to place parentheses.'''
        raise RegExException("Should have been filled in subclasses to determine parentheses")

    def _parenthesize(self, e: 'RegExTerm')->str:
        '''Convert to string with the right parentheses.'''
        if e._bindingLevel() < self._bindingLevel():
            return '('+str(e)+')'
        return str(e)

    def __str__(self)->str:
        raise RegExException("to be filled in subclasses")
 
    def simplify(self)->'RegExTerm':
        '''Return a simplified expression.'''
        return self

    def isEmptyWord(self)->bool:
        '''Return whether the expression represents the empty word.'''
        return False

    def isEmptySet(self)->bool:
        '''Return whether the expression represents the empty language.'''
        return False

    def asNBA(self)->Automaton:
        '''Converts omega-regular expression to an equivalent NBA.'''
        if not self.isOmegaRegEx():
            raise Exception('Not an omega-regular expression.')
        return self._asNBA()

    def asFSA(self)->Automaton:
        '''Converts finite regular expression to an equivalent FSA.'''
        if not self.isFiniteRegEx():
            raise Exception('Not a finite regular expression.')
        return self._asFSA()

    def _asNBA(self)->Automaton:
        '''Perform conversion to an NBA.'''
        raise RegExException("Overwrite in subclasses")

    def _asFSA(self)->Automaton:
        '''Perform conversion to an FSA.'''
        raise RegExException("Overwrite in subclasses")


    @staticmethod
    def fromFSA(A: Automaton)->'RegExTerm':

        # map a vertex to a tuple
        # - set of vertices with backward transition, 
        # - set of vertices with forward transition, 
        # - a dict with for every forward next vertex, a set of regular expressions with which the transition is labelled
        verticesMap: Dict[str,Tuple[Set[str],Set[str],Dict[str,Set['RegExTerm']]]]

        def _addTransition(u: str, v: str, re: 'RegExTerm'):
            '''Add (to) transition u->v if it exists, or create transition.'''
            nonlocal verticesMap
            if not v in verticesMap[u][1]:
                # add the forward link
                verticesMap[u][1].add(v)
                # add the backward link
                verticesMap[v][0].add(u)
                # create a set for labels
                verticesMap[u][2][v] = set()
            # add the regex to the transition labels
            verticesMap[u][2][v].add(re)

        def _removeTransition(u,v):
            '''remove transition u->v'''
            nonlocal verticesMap
            # Remove the forward link
            verticesMap[u][1].remove(v)
            # remove the regex labels
            del verticesMap[u][2][v]
            # remove the backward link
            verticesMap[v][0].remove(u)

        def _removeCycle(s)->Optional['RegExTerm']:
            '''If s has a cycle, return a regular expression that represents the alternatives of the labels on the cycle. Returns None if there is no cycle on s.'''
            nonlocal verticesMap
            # if s has transition to itself
            if s in verticesMap[s][1]:
                # create an alternatives term with each of the labels on the cycle
                re = RegExTermAlternatives(list(verticesMap[s][2][s]))
                # remove the cycle forward and backward links and labels
                verticesMap[s][0].remove(s)
                verticesMap[s][1].remove(s)
                del verticesMap[s][2][s]
                # return the alternatives expression created from the cycle
                return re
            else:
                # there is no cycle
                return None

        # create a graph 
        states: List[str] = A.statesInBFSOrder()
        verticesMap = dict()
        
        # add all states
        for s in A.states():
            # incoming states, outgoing states, map of outgoing states to sets of alternative REs
            verticesMap[s] = (set(), set(), dict())
        
        # add all edges
        for t in A.transitions():
            _addTransition(t[0], t[2], RegExTermLetter(t[1]))
        for t in A.epsilonTransitions():
            _addTransition(t[0], t[1], RegExTermEmptyWord())

        # make a single source connected to all initial states
        if '_src' in states:
            raise RegExException("Automaton has a clashing state name: '_src'")
        verticesMap['_src'] = (set(), set(), dict())
        for s in A.initialStates():
            _addTransition('_src', s, RegExTermEmptyWord())

        # make a single sink, connected to all final states
        if '_snk' in states:
            raise RegExException("Automaton has a clashing state name: '_snk'")
        verticesMap['_snk'] = (set(), set(), dict())
        for s in A.finalStates():
            _addTransition(s, '_snk', RegExTermEmptyWord())

        # eliminate all nodes for all states of the FSA by bypassing pairs of incoming and outgoing edges
        for s in states:
            # eliminate state s
            # incorporating self-loops
            cycleRegEx = _removeCycle(s)
            # for all pairs of incoming and outgoing edges make shortcut
            sIn = verticesMap[s][0].copy()
            sOut = verticesMap[s][1].copy()
            for u in sIn:
                for v in sOut:
                    re1 = RegExTermAlternatives(list(verticesMap[u][2][s]))
                    re2 = RegExTermAlternatives(list(verticesMap[s][2][v]))
                    if cycleRegEx:
                        reNew = RegExTermConcatenation([re1, RegExTermKleene(cycleRegEx), re2])
                    else:
                        reNew = RegExTermConcatenation([re1, re2])
                    # add (to) transition u->v
                    _addTransition(u, v, reNew)
            for u in sIn:
                # remove transition u->s
                _removeTransition(u, s)
            for v in sOut:
                # remove transition s->v
                _removeTransition(s, v)

        # check if there is a path from src to snk
        if not '_snk' in verticesMap['_src'][1]:
            return RegExTermEmptySet()
        return RegExTermAlternatives(list(verticesMap['_src'][2]['_snk']))

class RegExTermEmptySet(RegExTerm):

    def __init__(self):
        pass

    def _bindingLevel(self)->int:
        return 3

    def __str__(self)->str:
        return "\\o"

    def isEmptySet(self)->bool:
        return True

    def _asFSA(self)->Automaton:
        result = Automaton()
        si = result.addStateUnique("S")
        result.makeInitialState(si)
        return result

    def _asNBA(self)->Automaton:
        return self._asFSA()

class RegExTermEmptyWord(RegExTerm):

    def __init__(self):
        pass

    def _bindingLevel(self)->int:
        return 3

    def isOmegaRegEx(self)->bool:
        return False

    def __str__(self)->str:
        return "\\e"

    def isEmptyWord(self)->bool:
        return True

    def _asFSA(self)->Automaton:
        result = Automaton()
        si = result.addStateUnique("S")
        result.makeInitialState(si)
        result.makeFinalState(si)
        return result

class RegExTermConcatenation(RegExTerm):
    
    _expressions: List[RegExTerm]

    def __init__(self, expressions: List[RegExTerm]):
        self._expressions = expressions

    def subExpressions(self)->List[RegExTerm]:
        return self._expressions

    def isFiniteRegEx(self)->bool:
        for k in range(len(self._expressions)-1):
            if not self._expressions[k].isFiniteRegEx():
                raise Exception('Invalid regular expression.')
        if len(self._expressions) == 0:
            return True
        return self._expressions[-1].isFiniteRegEx()

    def isOmegaRegEx(self)->bool:
        for k in range(len(self._expressions)-1):
            if not self._expressions[k].isFiniteRegEx():
                raise Exception('Invalid regular expression.')
        if len(self._expressions) == 0:
            return True
        return self._expressions[-1].isOmegaRegEx()

    def simplify(self)->RegExTerm:

        simplifiedExpressions = [e.simplify() for e in self._expressions]
        if RegExTermEmptySet() in simplifiedExpressions:
            return RegExTermEmptySet()
        nonTrivialExpr = list(filter(lambda e: not e.isEmptyWord(), simplifiedExpressions))
        if len(nonTrivialExpr) == 0: return RegExTermEmptyWord()
        if len(nonTrivialExpr) == 1:
            return next(iter(nonTrivialExpr))
        return RegExTermConcatenation(nonTrivialExpr)


    @staticmethod
    def fromString(s: str)->Tuple[RegExTerm,str]:
        # read subexpressions until the terminating ')' symbol
        s = s[2:]
        expressions = list()
        while not re.match(r"\)", s):
            (exp, s) = RegExTerm.fromString(s)
            expressions.append(exp)
        return (RegExTermConcatenation(expressions), s[1:])

    def _bindingLevel(self)->int:
        return 1

    def __str__(self)->str:
        return ".".join([self._parenthesize(e) for e in self._expressions])

    def _asFSA(self)->Automaton:
        # get automata for subexpressions
        exprFSA = [e._asFSA() for e in self._expressions]
        ia = exprFSA[0]
        fa = exprFSA[-1]

        # build a new automaton
        result = Automaton()
        stateMap = dict()
        # add all states 
        for a in exprFSA:
            stateMap[a] = dict()
            for s in a.states():
                ns = result.addStateUnique(s)
                stateMap[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symbol, stateMap[a][dst])
            # add all epsilon transitions
            for (src, dst) in a.epsilonTransitions():
                result.addEpsilonTransition(stateMap[a][src], stateMap[a][dst])
        # add initial states
        for s in ia.initialStates():
            result.makeInitialState(stateMap[ia][s])
        # add final states
        for s in fa.finalStates():
            result.makeFinalState(stateMap[fa][s])
        # connect automata, n to n+1 while n+1 < len exprFSA
        n = 0
        while n+1 < len(exprFSA):
            aa = exprFSA[n]
            ab = exprFSA[n+1]
            for s in aa.finalStates():
                for t in ab.initialStates():
                    result.addEpsilonTransition(stateMap[aa][s], stateMap[ab][t])
            n += 1

        return result

    def _asNBA(self)->Automaton:
        # get automata for subexpressions
        exprFSA = [e._asFSA() for e in self._expressions[:-1]]
        if len(exprFSA) > 0:
            exprNBA = self._expressions[-1]._asNBA()
            exprFSA.append(exprNBA)

        ia = exprFSA[0]
        fa = exprFSA[-1]

        # build a new automaton
        result = Automaton()
        stateMap = dict()
        # add all states 
        for a in exprFSA:
            stateMap[a] = dict()
            for s in a.states():
                ns = result.addStateUnique(s)
                stateMap[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symbol, stateMap[a][dst])
            # add all epsilon transitions
            for (src, dst) in a.epsilonTransitions():
                result.addEpsilonTransition(stateMap[a][src], stateMap[a][dst])
        # add initial states
        for s in ia.initialStates():
            result.makeInitialState(stateMap[ia][s])
        # add final states
        for s in fa.finalStates():
            result.makeFinalState(stateMap[fa][s])
        # connect automata, n to n+1 while n+1 < len exprFSA
        n = 0
        while n+1 < len(exprFSA):
            aa = exprFSA[n]
            ab = exprFSA[n+1]
            for s in aa.finalStates():
                for t in ab.initialStates():
                    result.addEpsilonTransition(stateMap[aa][s], stateMap[ab][t])
            n += 1

        return result

class RegExTermAlternatives(RegExTerm):

    _expressions: List[RegExTerm]
    
    def __init__(self, expressions: List[RegExTerm]):
        self._expressions = expressions

    def subExpressions(self)->List[RegExTerm]:
        return self._expressions

    def isFiniteRegEx(self)->bool:
        for e in self._expressions:
            if not e.isFiniteRegEx():
                return False
        return True

    def isOmegaRegEx(self)->bool:
        for e in self._expressions:
            if not e.isOmegaRegEx():
                return False
        return True

    def simplify(self)->RegExTerm:
        simplifiedExpressions = [e.simplify() for e in self._expressions]
        nonTrivialExpr = list(filter(lambda e: not e.isEmptySet(), simplifiedExpressions))
        if len(nonTrivialExpr) == 0: return RegExTermEmptySet()
        uniqueExpressions = set(nonTrivialExpr)
        if len(uniqueExpressions) == 1:
            return next(iter(uniqueExpressions))
        return RegExTermAlternatives(list(uniqueExpressions))

    @staticmethod
    def fromString(s: str)->Tuple[RegExTerm,str]:
        # read subexpressions until the terminating ')' symbol
        s = s[2:]
        expressions = list()
        while not re.match(r"\)", s):
            (exp, s) = RegExTerm.fromString(s)
            expressions.append(exp)
        return (RegExTermAlternatives(expressions), s[1:])

    def _bindingLevel(self)->int:
        return 0

    def __str__(self)->str:
        return "+".join([self._parenthesize(e) for e in self._expressions])

    def _asFSA(self)->Automaton:
        # get automata for subexpressions
        exprFSA = [e._asFSA() for e in self._expressions]

        # build a new automaton
        result = Automaton()
        stateMap = dict()
        # add all states 
        for a in exprFSA:
            stateMap[a] = dict()
            for s in a.states():
                ns = result.addStateUnique(s)
                stateMap[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symbol, stateMap[a][dst])
            # add all epsilon transitions
            for (src, dst) in a.epsilonTransitions():
                result.addEpsilonTransition(stateMap[a][src], stateMap[a][dst])
            # add initial states
            for s in a.initialStates():
                result.makeInitialState(stateMap[a][s])
            # add final states
            for s in a.finalStates():
                result.makeFinalState(stateMap[a][s])

        return result

    def _asNBA(self)->Automaton:
        # get automata for subexpressions
        exprNBA = [e.asNBA() for e in self._expressions]

        # build a new automaton
        result = Automaton()
        stateMap = dict()
        # add all states 
        for a in exprNBA:
            stateMap[a] = dict()
            for s in a.states():
                ns = result.addStateUnique(s)
                stateMap[a][s] = ns
            # add all transitions
            for (src, symbol, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symbol, stateMap[a][dst])
            # add all epsilon transitions
            for (src, dst) in a.epsilonTransitions():
                result.addEpsilonTransition(stateMap[a][src], stateMap[a][dst])
            # add initial states
            for s in a.initialStates():
                result.makeInitialState(stateMap[a][s])
            # add final states
            for s in a.finalStates():
                result.makeFinalState(stateMap[a][s])

        return result

class RegExTermKleene(RegExTerm):

    _expression: RegExTerm
    
    def __init__(self, expression: RegExTerm):
        self._expression = expression

    def isFiniteRegEx(self)->bool:
        if not self._expression.isFiniteRegEx():
            raise Exception('Illegal regular expression.')
        return True

    def isOmegaRegEx(self)->bool:
        return False

    def simplify(self)->RegExTerm:
        simplifiedExpression = self._expression.simplify()
        if simplifiedExpression.isEmptySet():
            return RegExTermEmptySet()
        if simplifiedExpression.isEmptyWord():
            return RegExTermEmptyWord()
        return RegExTermKleene(simplifiedExpression)

    def subExpressions(self)->Set[RegExTerm]:
        res = set()
        res.add(self._expression)
        return res

    @staticmethod
    def fromString(s: str)->Tuple[RegExTerm,str]:
        # read subexpression and the terminating ')' symbol
        s = s[2:]
        (exp, s) = RegExTerm.fromString(s)
        return (RegExTermKleene(exp), s[1:])

    def _bindingLevel(self)->int:
        return 2

    def __str__(self)->str:
        return self._parenthesize(self._expression)+"*"

    def _asFSA(self)->Automaton:
        # get automaton for subexpression
        result = self._expression._asFSA()

        # add a new state that is initial and final
        sif = result.addStateUnique("S")

        # add feedback transitions
        for s in result.initialStates():
            result.addEpsilonTransition(sif, s)
        for s in result.finalStates():
            result.addEpsilonTransition(s, sif)

        # make the new state the only initial state
        result.clearInitialStates()
        result.makeInitialState(sif)
        result.clearFinalStates()
        result.makeFinalState(sif)

        return result
class RegExTermOmega(RegExTerm):

    _expression: RegExTerm
    
    def __init__(self, expression: RegExTerm):
        self._expression = expression

    def isFiniteRegEx(self)->bool:
        return False

    def isOmegaRegEx(self)->bool:
        if not self._expression.isFiniteRegEx():
            raise Exception('Illegal regular expression.')
        return True

    def simplify(self)->RegExTerm:
        simplifiedExpression = self._expression.simplify()
        if simplifiedExpression.isEmptySet():
            return RegExTermEmptySet()
        if simplifiedExpression.isEmptyWord():
            raise Exception('Invalid omega-regular expression')
        return RegExTermOmega(simplifiedExpression)

    def subExpressions(self)->Set[RegExTerm]:
        res = set()
        res.add(self._expression)
        return res

    @staticmethod
    def fromString(s: str)->Tuple[RegExTerm,str]:
        # read subexpression and the terminating ')' symbol
        s = s[2:]
        (exp, s) = RegExTerm.fromString(s)
        return (RegExTermOmega(exp), s[1:])

    def _bindingLevel(self)->int:
        return 2

    def __str__(self)->str:
        return self._parenthesize(self._expression)+"**"

    def _asNBA(self)->Automaton:
        # get automaton for subexpression
        result = self._expression._asFSA()

        # add feedback transitions
        for s in result.initialStates():
            for t in result.finalStates():
                result.addEpsilonTransition(t, s)

        return result

class RegExTermLetter(RegExTerm):
    
    _letter: str
    
    def __init__(self, letter: str):
        self._letter = letter

    def isOmegaRegEx(self)->bool:
        return False

    def _collectAlphabet(self, result: Set[str])->None:
        result.add(self._letter)

    @staticmethod
    def fromString(s: str)->Tuple[RegExTerm,str]:
        m = REGEXLETTER.search(s)
        if m is None:
            raise RegExException("Failed to match REGEXLETTER expression.")
        letter = m.group(0)
        return (RegExTermLetter(letter), s[len(letter):])

    def _bindingLevel(self)->int:
        return 3

    def __str__(self)->str:
        if REGEXLETTERSIMPLE.match(self._letter):
            return self._letter
        return "'"+self._letter.replace("'", "\\'")+"'"

    def _asFSA(self)->Automaton:
        
        result = Automaton()
        si = result.addStateUnique("S")
        sf = result.addStateUnique("S")
        symbol = self._letter.replace("'", "")
        result.addTransition(si, symbol, sf)
        result.makeInitialState(si)
        result.makeFinalState(sf)
        return result

class RegEx(object):

    _expression: RegExTerm
    _name: str

    def __init__(self, name: str, expression: RegExTerm):
        self._expression = expression
        self._name = name

    def asFSA(self)->Automaton:
        if not self._expression.isFiniteRegEx():
            raise Exception('Not a finite regular expression.')
        return self._expression.asFSA()

    def alphabet(self)->Set[str]:
        '''Return the alphabet of the regex '''
        result: Set[str] = set()
        self._expression._collectAlphabet(result)
        return result

    def isOmegaRegEx(self)->bool:
        return self._expression.isOmegaRegEx()

    def isFiniteRegEx(self)->bool:
        return self._expression.isFiniteRegEx()

    def asDSL(self, name: str):
        return 'regular expression {} = {}'.format(name, str(self))

    @staticmethod
    def fromDSL(regexString)->Tuple[str,RegExTerm]:
        factory = dict()
        factory['Letter'] = lambda l: RegExTermLetter(l)
        factory['Kleene'] = lambda exp: RegExTermKleene(exp)
        factory['Omega'] = lambda exp: RegExTermOmega(exp)
        factory['Alternatives'] = lambda exp: RegExTermAlternatives(exp)
        factory['Concatenations'] = lambda exp: RegExTermConcatenation(exp)
        factory['EmptyLanguage'] = lambda : RegExTermEmptySet()
        factory['EmptyWord'] = lambda : RegExTermEmptyWord()
        (name, expression) = parseRegExDSL(regexString, factory)
        if name is None or expression is None:
            exit(1)
        return name, expression

    @staticmethod
    def fromString(regexString):

        # find the name
        match = re.search(r".*regular\s+expression\s+(?P<name>[^\s]+)\s*=\s*(?P<regex>[^\s]*?)\s*$", regexString, re.MULTILINE|re.DOTALL)
        if match is None:
            raise Exception("Input is not a valid RegEx")
        name = match.group('name')

        (regex, _) = RegExTerm.fromString(match.group('regex'))
  
        return (name, RegEx(name, regex))

    @staticmethod
    def fromFSA(A, name):
        return RegEx(name, RegExTerm.fromFSA(A).simplify())

    def __str__(self)->str:
        return str(self._expression)
