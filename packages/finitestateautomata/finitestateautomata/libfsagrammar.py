'''DSL grammar parsing support for finite state automata.'''

import sys
from typing import Any, Optional, Tuple
from textx import TextXSyntaxError, metamodel_from_str

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\FiniteStateAutomata.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


FSA_GRAMMAR = """
FiniteStateAutomatonModel:
	('author' '=' author=ID)?
	'finite' 'state' 'automaton' name=ID '{'
	edges += Edge*
	('states' states += State*)?
	'}'
;

Edge:
		src_state=State
        (( ('-')+ '>' dst_state=State) | ( ('-')+ (specs=EdgeSpecs) ('-')+ '>' dst_state=State))
;

EdgeSpecs:
	annotations += EdgeAnnotation ((','|';') annotations += EdgeAnnotation)*

;

EdgeAnnotation:
	symbol= ID | symbol=STRING | symbol=EPSILON_SYMBOL
;

State:
	u_state=UndecoratedState (specs=StateSpecs)?
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
	(initialOrFinal = INITIAL_OR_FINAL)  ('[' (acceptanceSets += ID) (',' (acceptanceSets += ID))* ']')?
;

INITIAL_OR_FINAL:
    'final' | 'f' | 'initial' | 'i'
;


StateName: ID;

Number:
	Float | INT
;

Float: INT '.' INT;

EPSILON_SYMBOL:
	'#'
;

Comment:
    /\\/\\*(.|\\n)*?\\*\\// | /\\/\\/.*?$/
;


"""

MetaModelFSA = metamodel_from_str(FSA_GRAMMAR, classes=[])

def parse_fsa_dsl(content: str, factory: Any)->Tuple[Optional[str], Optional[Any]]:
    '''Parse Finite State Automaton from DSL string.'''
    try:
        model =  MetaModelFSA.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write(f"Syntax error in line {err.line} col {err.col}: {err.message}")
        return (None, None)
    fsa = factory['Init']()
    for e in model.edges:
        _parse_edge(e, fsa, factory)
    for s in model.states:
        _parse_state(s, fsa, factory)

    return (model.name, fsa)

def _parse_edge(e, fsa, factory):
    src_state = _parse_state(e.src_state, fsa, factory)
    dst_state = _parse_state(e.dst_state, fsa, factory)
    if e.specs:
        for symbol in  _parse_edge_specs(e.specs):
            factory['addTransitionPossiblyEpsilon'](fsa, src_state, dst_state, symbol)
    else:
        factory['AddEpsilonTransition'](fsa, src_state, dst_state)

def _parse_state(s, fsa, factory):
    state = _parse_undecorated_state(s.u_state)
    labels, acceptance_sets = _parse_state_specs(s.specs)
    factory['AddState'](fsa, state, labels, acceptance_sets)
    return state

def _parse_edge_specs(specs):
    if not specs:
        return set()
    return [a.symbol for a in specs.annotations]

def _parse_undecorated_state(u_state):
    if u_state.name:
        return u_state.name
    if u_state.stateSet:
        return '{' + (','.join([_parse_undecorated_state(us) for us in \
                                u_state.stateSet.states])) + '}'
    return '(' + (','.join([_parse_undecorated_state(us) for us in \
                            u_state.stateTuple.states])) + ')'


def _parse_state_specs(specs):
    if not specs:
        return set(), set()
    labels = set()
    acceptance_sets = set()
    for a in specs.annotations:
        labels.add(a.initialOrFinal)
        if a.acceptanceSets:
            acceptance_sets.update(a.acceptanceSets)
    if len(acceptance_sets) == 0:
        acceptance_sets.add('default')
    return labels, acceptance_sets
