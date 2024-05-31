'''DSL grammar parsing support for linear temporal logic.'''

import sys
from typing import Any, Dict, Optional, Set, Tuple, Union
from textx import metamodel_from_str, TextXSyntaxError

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\LinearTemporalLogic.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


LTL_GRAMMAR = """

LTLModel:
	'ltl' 'formula' name=LTL_ID '='
	formula = LTLFormula
	('alphabet' alphabet = SetOfSymbols
	)?
	('where' (definitions = Definition )*
	)?
;

Definition:
	proposition = LTL_ID '=' symbols = SetOfSymbols
;

SetOfSymbols:
	'{' (symbols = LTL_ID) (',' symbols = LTL_ID)* '}'
;

LTLFormula:
	formula = LTLFormula1
	(
		'or'
		alternatives = LTLFormula1
		('or' alternatives = LTLFormula1 )*
	)?
;

LTLFormula1:
	formula = LTLFormula2
	(
		'and'
		alternatives = LTLFormula2
		('and' alternatives = LTLFormula2 )*
	)?
;


LTLFormula2:
		(sub_expression_1 = LTLFormula3) ('U' sub_expression_2 = LTLFormula2)?
;

LTLFormula3:
		(sub_expression_1 = LTLFormula4) ('R' sub_expression_2 = LTLFormula3)?
;

LTLFormula4:
		sub_expression = LTLFormula5 ('=>' consequence = LTLFormula4)?
;

LTLFormula5:
		('X' next_sub_expression = LTLFormula5) |
		('F' eventually_sub_expression = LTLFormula5) |
		('G' always_sub_expression = LTLFormula5) |
		('not' not_sub_expression = LTLFormula5) |
		sub_expression = LTLFormula6
;



LTLFormula6:
		trueExpression = True |
		falseExpression = False |
		propositionExpression = PropositionExpression |
		'(' expression = LTLFormula ')'
;


PropositionExpression:
	Proposition
;

Proposition:
	LTL_ID | STRING
;

LTL_ID:
	/[a-zA-Z][a-zA-Z0-9]*/
;
True:
    /true/
;

False:
    /false/
;

Comment:
    /\\/\\*(.|\\n)*?\\*\\// | /\\/\\/.*?$/
;

"""

MetaModelLTL = metamodel_from_str(LTL_GRAMMAR, classes=[])

def parse_ltl_dsl(content: str, factory: Any)->Union[Tuple[None,None],Tuple[str,Tuple[Any, \
                    Optional[Set[str]],Optional[Dict[str,Set[str]]]]]]:
    '''Parse LTL formula from DSL string.'''
    try:
        model =  MetaModelLTL.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write(f"Syntax error in line {err.line} col {err.col}: {err.message}")
        return (None, None)
    return (model.name, _parse_lt_model(model, factory))

def _parse_lt_model(model: Any, factory: Any)->Tuple[Any,Optional[Set[str]], \
                                                     Optional[Dict[str,Set[str]]]]:
    return (_parse_ltl_formula(model.formula, factory), _parse_alphabet(model.alphabet), \
            _parse_definitions(model.definitions))


def _parse_alphabet(alphabet: Any)->Optional[Set[str]]:
    if alphabet is None:
        return None
    return set(alphabet.symbols)

def _parse_definitions(definitions)->Optional[Dict[str,Set[str]]]:
    if len(definitions) == 0:
        return None
    defs = {}
    for d in definitions:
        defs[d.proposition] = set(d.symbols.symbols)
    return defs

def _parse_ltl_formula(f: Any, factory: Any)->Any:
    if len(f.alternatives) > 0:
        fs = [ _parse_ltl_formula_1(f.formula, factory) ]
        for phi in f.alternatives:
            fs.append(_parse_ltl_formula_1(phi, factory))
        return factory['Disjunction'](fs)
    return _parse_ltl_formula_1(f.formula, factory)

def _parse_ltl_formula_1(f: Any, factory: Any)->Any:
    if len(f.alternatives) > 0:
        fs = [ _parse_ltl_formula_2(f.formula, factory) ]
        for phi in f.alternatives:
            fs.append(_parse_ltl_formula_2(phi, factory))
        return factory['Conjunction'](fs)
    return _parse_ltl_formula_2(f.formula, factory)

def _parse_ltl_formula_2(f: Any, factory: Any)->Any:
    phi1 = _parse_ltl_formula_3(f.sub_expression_1, factory)
    if f.sub_expression_2:
        phi2 = _parse_ltl_formula_2(f.sub_expression_2, factory)
        return factory['Until'](phi1, phi2)
    return phi1

def _parse_ltl_formula_3(f: Any, factory: Any)->Any:
    phi1 = _parse_ltl_formula_4(f.sub_expression_1, factory)
    if f.sub_expression_2:
        phi2 = _parse_ltl_formula_3(f.sub_expression_2, factory)
        return factory['Release'](phi1, phi2)
    return phi1

def _parse_ltl_formula_4(f: Any, factory: Any)->Any:
    phi1 = _parse_ltl_formula_5(f.sub_expression, factory)
    if f.consequence:
        phi2 = _parse_ltl_formula_4(f.consequence, factory)
        return factory['Disjunction']([factory['Negation'](phi1), phi2])
    return phi1

def _parse_ltl_formula_5(f: Any, factory: Any)->Any:
    if f.next_sub_expression:
        return factory['Next'](_parse_ltl_formula_5(f.next_sub_expression, factory))
    if f.eventually_sub_expression:
        return factory['Eventually'](_parse_ltl_formula_5(f.eventually_sub_expression, factory))
    if f.always_sub_expression:
        return factory['Always'](_parse_ltl_formula_5(f.always_sub_expression, factory))
    if f.not_sub_expression:
        return factory['Negation'](_parse_ltl_formula_5(f.not_sub_expression, factory))
    return _parse_ltl_formula_6(f.sub_expression, factory)

def _parse_ltl_formula_6(f: Any, factory: Any)->Any:
    if f.trueExpression:
        return factory['True']()
    if f.falseExpression:
        return factory['False']()
    if f.propositionExpression:
        return factory['Proposition'](f.propositionExpression)
    else:
        return _parse_ltl_formula(f.expression, factory)
