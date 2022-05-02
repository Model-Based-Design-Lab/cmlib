from textx import metamodel_from_str, TextXSyntaxError
import sys

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\FiniteStateAutomata.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


FSAGrammar = """
FiniteStateAutomatonModel:
	('author' '=' author=ID)?
	'finite' 'state' 'automaton' name=ID '{'
	edges += Edge*
	('states' states += State*)?
	'}'
;

Edge:
		srcstate=State
        (( ('-')+ '>' dststate=State) | ( ('-')+ (specs=EdgeSpecs) ('-')+ '>' dststate=State))
;

EdgeSpecs:
	annotations += EdgeAnnotation ((','|';') annotations += EdgeAnnotation)*
	
;

EdgeAnnotation:
	symbol= ID | symbol=STRING | symbol=EPSILONSYMBOL
;

State:
	ustate=UndecoratedState (specs=StateSpecs)?
;


UndecoratedState:
		name=ID | 
		stateSet = StateSet | 
		stateTuple = StateTuple
;

StateSet:
	'{' (states += UndecoratedState) ((',') states += UndecoratedState)* '}'
;

StateTuple:
	'(' (states += UndecoratedState) ((',') states += UndecoratedState)* ')'
;


StateSpecs:
	('[' annotations += StateAnnotation (';' annotations += StateAnnotation)* ']') |
	(annotations += StateAnnotation (';' annotations += StateAnnotation)* )
;

StateAnnotation:
	(initialOrFinal = INITIALORFINAL)  ('[' (acceptanceSets += ID) (',' (acceptanceSets += ID))* ']')?
;

INITIALORFINAL:
    'final' | 'f' | 'initial' | 'i'
;


StateName: ID;

Number:
	Float | INT
;

Float: INT '.' INT;

EPSILONSYMBOL:
	'#'
;

Comment:
    /\\/\\*(.|\\n)*?\\*\\// | /\\/\\/.*?$/
;


"""

MetaModelFSA = metamodel_from_str(FSAGrammar, classes=[])

def parseFSADSL(content, factory):
    try:
        model =  MetaModelFSA.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write("Syntax error in line %d col %d: %s" % (err.line, err.col, err.message))
        return (None, None)
    fsa = factory['Init']()
    for e in model.edges:
        parseEdge(e, fsa, factory)
    for s in model.states:
        parseState(s, fsa, factory) 

    return (model.name, fsa)

def parseEdge(e, fsa, factory):
    srcState = parseState(e.srcstate, fsa, factory)
    dstState = parseState(e.dststate, fsa, factory)
    if e.specs:
        for symb in  parseEdgeSpecs(e.specs):
            factory['addTransitionPossiblyEpsilon'](fsa, srcState, dstState, symb)
    else:
        factory['AddEpsilonTransition'](fsa, srcState, dstState)
    
def parseState(s, fsa, factory):
    state = parseUndecoratedState(s.ustate)
    labels, acceptanceSets = parseStateSpecs(s.specs)
    factory['AddState'](fsa, state, labels, acceptanceSets)
    return state

def parseEdgeSpecs(specs):
    if not specs:
        return set()
    return [a.symbol for a in specs.annotations]

def parseUndecoratedState(ustate):
    if ustate.name:
        return ustate.name
    if ustate.stateSet:
        return '{' + (','.join([parseUndecoratedState(us) for us in ustate.stateSet.states])) + '}'
    return '(' + (','.join([parseUndecoratedState(us) for us in ustate.stateTuple.states])) + ')'


def parseStateSpecs(specs):
    if not specs:
        return set(), set()
    labels = set()
    acceptanceSets = set()
    for a in specs.annotations:
        labels.add(a.initialOrFinal)
        if a.acceptanceSets:
            acceptanceSets.update(a.acceptanceSets)
    if len(acceptanceSets) == 0:
        acceptanceSets.add('default')
    return labels, acceptanceSets
    