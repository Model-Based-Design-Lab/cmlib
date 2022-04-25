from functools import reduce
from io import StringIO
from enum import Enum
import re

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

class RegExTerm(object):

    def __init__(self, exprType, subexpressions = list()):
        pass

    # parse expression. return expression and remainder of the string
    @staticmethod
    def fromString(s):
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
        raise Exception('malformed reglar expression')

    # parse expression. return expression and remainder of the string
    @staticmethod
    def fromDSL(s):
        pass


    def subExpressions(self):
        return set()

    def _collectAlphabet(self, result):
        for e in self.subExpressions():
            e._collectAlphabet(result)

    def isFiniteRegEx(self):
        # default result, override if necessary
        return True

    def isOmegaRegEx(self):
        # default result, override if necessary
        return True

    def _bindingLevel(self):
        return "to be filled in subclasses to determine parentheses"

    def _parenthesize(self, e):
        if e._bindingLevel() < self._bindingLevel():
            return '('+str(e)+')'
        return str(e)

    def __str__(self):
        return "to be filled in subclasses"

 
    def simplify(self):
        return self

    def isEmptyWord(self):
        return False

    def isEmptySet(self):
        return False

    def asNBA(self):
        if not self.isOmegaRegEx():
            raise Exception('Not an omega-regular expression.')
        return self._asNBA()

    def asFSA(self):
        if not self.isFiniteRegEx():
            raise Exception('Not a finite regular expression.')
        return self._asFSA()

    def _asNBA(self):
        pass

    def _asFSA(self):
        pass


    @staticmethod
    def fromFSA(A):

        def _addTransition(u, v, re):
            # add (to) transition u->v if it exists, or create transition
            if not v in verticesMap[u][1]:
                verticesMap[u][1].add(v)
                verticesMap[u][2][v] = set()
                verticesMap[v][0].add(u)
            verticesMap[u][2][v].add(re)

        def _removeTransition(u,v):
            # remove transition u->v
            verticesMap[u][1].remove(v)
            del verticesMap[u][2][v]
            verticesMap[v][0].remove(u)


        def _removeCycle(s):
            if s in verticesMap[s][1]:
                re = RegExTermAlternatives(list(verticesMap[s][2][s]))
                verticesMap[s][0].remove(s)
                verticesMap[s][1].remove(s)
                del verticesMap[s][2][s]
                return re
            else:
                # there is no cycle
                return None

        # # take care of the degenerate case
        # if len(A.initialStates()) == 0 or len(A.finalStates()) == 0:
        #     return RegExTermEmptySet()

        # create a graph 
        states = A.statesInBFSOrder()
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
        verticesMap['_src'] = (set(), set(), dict())
        for s in A.initialStates():
            _addTransition('_src', s, RegExTermEmptyWord())

        # make a single sink, connected to all final states
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
                    re1 = RegExTermAlternatives(verticesMap[u][2][s])
                    re2 = RegExTermAlternatives(verticesMap[s][2][v])
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

    def _bindingLevel(self):
        return 3

    def __str__(self):
        return "\\o"

    def isEmptySet(self):
        return True

    def _asFSA(self):
        result = Automaton()
        si = result.addStateUnique("S")
        result.makeInitialState(si)
        return result

    def _asNBA(self):
        return self._asFSA()

class RegExTermEmptyWord(RegExTerm):

    def __init__(self):
        pass

    def _bindingLevel(self):
        return 3

    def isOmegaRegEx(self):
        return False

    def __str__(self):
        return "\\e"

    def isEmptyWord(self):
        return True

    def _asFSA(self):
        result = Automaton()
        si = result.addStateUnique("S")
        result.makeInitialState(si)
        result.makeFinalState(si)
        return result

class RegExTermConcatenation(RegExTerm):

    def __init__(self, expressions):
        self._expressions = expressions

    def subExpressions(self):
        return self._expressions

    def isFiniteRegEx(self):
        for k in range(len(self._expressions)-1):
            if not self._expressions[k].isFiniteRegEx():
                raise Exception('Invalid regular expression.')
        if len(self._expressions) == 0:
            return True
        return self._expressions[-1].isFiniteRegEx()

    def isOmegaRegEx(self):
        for k in range(len(self._expressions)-1):
            if not self._expressions[k].isFiniteRegEx():
                raise Exception('Invalid regular expression.')
        if len(self._expressions) == 0:
            return True
        return self._expressions[-1].isOmegaRegEx()

    def simplify(self):

        simplifiedExpressions = [e.simplify() for e in self._expressions]
        if RegExTermEmptySet() in simplifiedExpressions:
            return RegExTermEmptySet()
        nonTrivialExpr = list(filter(lambda e: not e.isEmptyWord(), simplifiedExpressions))
        if len(nonTrivialExpr) == 0: return RegExTermEmptyWord()
        if len(nonTrivialExpr) == 1:
            return next(iter(nonTrivialExpr))
        return RegExTermConcatenation(nonTrivialExpr)


    @staticmethod
    def fromString(s):
        # read subexpressions until the terminating ')' symbol
        s = s[2:]
        expressions = list()
        while not re.match(r"\)", s):
            (exp, s) = RegExTerm.fromString(s)
            expressions.append(exp)
        return (RegExTermConcatenation(expressions), s[1:])

    def _bindingLevel(self):
        return 1

    def __str__(self):
        return ".".join([self._parenthesize(e) for e in self._expressions])

    def _asFSA(self):
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
            for (src, symb, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symb, stateMap[a][dst])
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

    def _asNBA(self):
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
            for (src, symb, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symb, stateMap[a][dst])
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

    def __init__(self, expressions):
        self._expressions = expressions

    def subExpressions(self):
        return self._expressions

    def isFiniteRegEx(self):
        for e in self._expressions:
            if not e.isFiniteRegEx():
                return False
        return True

    def isOmegaRegEx(self):
        for e in self._expressions:
            if not e.isOmegaRegEx():
                return False
        return True

    def simplify(self):
        simplifiedExpressions = [e.simplify() for e in self._expressions]
        nonTrivialExpr = list(filter(lambda e: not e.isEmptySet(), simplifiedExpressions))
        if len(nonTrivialExpr) == 0: return RegExTermEmptySet()
        uniqueExpressions = set(nonTrivialExpr)
        if len(uniqueExpressions) == 1:
            return next(iter(uniqueExpressions))
        return RegExTermAlternatives(list(uniqueExpressions))

    @staticmethod
    def fromString(s):
        # read subexpressions until the terminating ')' symbol
        s = s[2:]
        expressions = list()
        while not re.match(r"\)", s):
            (exp, s) = RegExTerm.fromString(s)
            expressions.append(exp)
        return (RegExTermAlternatives(expressions), s[1:])

    def _bindingLevel(self):
        return 0

    def __str__(self):
        return "+".join([self._parenthesize(e) for e in self._expressions])

    def _asFSA(self):
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
            for (src, symb, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symb, stateMap[a][dst])
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

    def _asNBA(self):
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
            for (src, symb, dst) in a.transitions():
                result.addTransition(stateMap[a][src], symb, stateMap[a][dst])
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

    def __init__(self, expression):
        self._expression = expression

    def isFiniteRegEx(self):
        if not self._expression.isFiniteRegEx():
            raise Exception('Illegal regular expression.')
        return True

    def isOmegaRegEx(self):
        return False

    def simplify(self):
        simplifiedExpression = self._expression.simplify()
        if simplifiedExpression.isEmptySet():
            return RegExTermEmptySet()
        if simplifiedExpression.isEmptyWord():
            return RegExTermEmptyWord()
        return RegExTermKleene(simplifiedExpression)

    def subExpressions(self):
        res = set()
        res.add(self._expression)
        return res

    @staticmethod
    def fromString(s):
        # read subexpression and the terminating ')' symbol
        s = s[2:]
        (exp, s) = RegExTerm.fromString(s)
        return (RegExTermKleene(exp), s[1:])

    def _bindingLevel(self):
        return 2

    def __str__(self):
        return self._parenthesize(self._expression)+"*"

    def _asFSA(self):
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

    def __init__(self, expression):
        self._expression = expression

    def isFiniteRegEx(self):
        return False

    def isOmegaRegEx(self):
        if not self._expression.isFiniteRegEx():
            raise Exception('Illegal regular expression.')
        return True

    def simplify(self):
        simplifiedExpression = self._expression.simplify()
        if simplifiedExpression.isEmptySet():
            return RegExTermEmptySet()
        if simplifiedExpression.isEmptyWord():
            raise Exception('Invalid omega-regular expression')
        return RegExTermOmega(simplifiedExpression)

    def subExpressions(self):
        res = set()
        res.add(self._expression)
        return res

    @staticmethod
    def fromString(s):
        # read subexpression and the terminating ')' symbol
        s = s[2:]
        (exp, s) = RegExTerm.fromString(s)
        return (RegExTermOmega(exp), s[1:])

    def _bindingLevel(self):
        return 2

    def __str__(self):
        return self._parenthesize(self._expression)+"**"

    def _asNBA(self):
        # get automaton for subexpression
        result = self._expression._asFSA()

        # add feedback transitions
        for s in result.initialStates():
            for t in result.finalStates():
                result.addEpsilonTransition(t, s)

        return result

class RegExTermLetter(RegExTerm):
    def __init__(self, letter):
        self._letter = letter

    def isOmegaRegEx(self):
        return False

    def _collectAlphabet(self, result):
        result.add(self._letter)

    @staticmethod
    def fromString(s):
        letter = REGEXLETTER.search(s).group(0)
        return (RegExTermLetter(letter), s[len(letter):])

    def _bindingLevel(self):
        return 3

    def __str__(self):
        if REGEXLETTERSIMPLE.match(self._letter):
            return self._letter
        return "'"+self._letter.replace("'", "\\'")+"'"

    def _asFSA(self):
        
        result = Automaton()
        si = result.addStateUnique("S")
        sf = result.addStateUnique("S")
        symb = self._letter.replace("'", "")
        result.addTransition(si, symb, sf)
        result.makeInitialState(si)
        result.makeFinalState(sf)
        return result

class RegEx(object):

    def __init__(self, name, expression):
        self._expression = expression
        self._name = name

    def asFSA(self):
        if not self._expression.isFiniteRegEx():
            raise Exception('Not a finite regular expression.')
        return self._expression.asFSA()

    def alphabet(self):
        '''Return the alphabet of the regex '''
        return self._expression._collectAlphabet(set())

    def isOmegaRegEx(self):
        return self._expression.isOmegaRegEx()

    def isFiniteRegEx(self):
        return self._expression.isFiniteRegEx()

    def asDSL(self, name):
        return 'regular expression {} = {}'.format(name, str(self))

    @staticmethod
    def fromDSL(regexString):
        factory = dict()
        factory['Letter'] = lambda l: RegExTermLetter(l)
        factory['Kleene'] = lambda exp: RegExTermKleene(exp)
        factory['Omega'] = lambda exp: RegExTermOmega(exp)
        factory['Alternatives'] = lambda exp: RegExTermAlternatives(exp)
        factory['Concatenations'] = lambda exp: RegExTermConcatenation(exp)
        factory['EmptyLanguage'] = lambda : RegExTermEmptySet()
        factory['EmptyWord'] = lambda : RegExTermEmptyWord()
        res = parseRegExDSL(regexString, factory)
        if res[0] is None:
            exit(1)
        return res

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

    def __str__(self):
        return str(self._expression)
