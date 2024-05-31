'''DSL grammar parsing support for regular expressions.'''

import sys
from typing import Any, Dict, Optional, Tuple
from textx import TextXSyntaxError, metamodel_from_str

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\RegularExpressions.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


REG_EX_GRAMMAR = """
RegExModel:
	'regular' 'expression' name=ID '='
	expression = RegularExpression
	('where' (definitions = Definition )*
	)?
;
Definition:
	symbol = ID '=' expression = RegularExpression
;
RegularExpression:
	expression = RegularExpression1
	(
		'+'
		alternatives = RegularExpression1
		('+' alternatives = RegularExpression1 )*
	)?
;
RegularExpression1:
	expression = RegularExpression2
	(
		'.'
		concatenations = RegularExpression2
		('.' concatenations = RegularExpression2)*
 	)?
;
RegularExpression2:
		subexpression = RegularExpression3
		(
			(omega = '**')|
			(kleene = '*')
		)?
;


RegularExpression3:
		emptyLangExpression = EmptyLanguageExpression |
		emptyWordExpression = EmptyWordExpression |
		letterExpression = LetterExpression |
		referenceExpression = ReferenceExpression |
		'(' expression = RegularExpression ')'
;
EmptyLanguageExpression:
	EMPTYSET
;
EmptyWordExpression:
	EPSILON
;
LetterExpression:
	Letter
;
ReferenceExpression:
	'@' reference = ID
;
Letter:
	SIMPLE_LETTER | STRING
;
SIMPLE_LETTER:
	/[a-zA-Z]/
;

EMPTYSET:
	'\\o'
;

EPSILON:
	'\\e'
;
Comment:
    /\\/\\*(.|\\n)*?\\*\\// | /\\/\\/.*?$/
;


"""

MetaModelRegEx = metamodel_from_str(REG_EX_GRAMMAR, classes=[])

def parse_reg_ex_dsl(content: str, factory: Any)->Tuple[Optional[str],Optional[Any]]:
    '''Parse string to regular expression model.'''
    try:
        model =  MetaModelRegEx.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write(f"Syntax error in line {err.line} col {err.col}: {err.message}")
        return (None, None)
    regex = _parse_refs_and_regular_expression(model, factory)
    return (model.name, regex)

def _parse_refs_and_regular_expression(m: Any, factory: Any)->Any:
    references = {}
    if m.definitions:
        references = _parse_refs(m.definitions)
    return _parse_regular_expression(m.expression, references, factory)

def _parse_refs(defs:Dict[Any,Any])->Dict[str,Any]:
    res:Dict[str,Any] = {}
    res['_processed'] = set()
    for d in defs:
        res[d.symbol] = d.expression
    return res

def _parse_regular_expression(m: Any, references:Dict[str,Any], factory: Any)->Any:
    if len(m.alternatives) > 0:
        expr = [ _parse_regular_expression1(m.expression, references, factory) ]
        for n in m.alternatives:
            expr.append(_parse_regular_expression1(n, references, factory))
        return factory['Alternatives'](expr)
    return _parse_regular_expression1(m.expression, references, factory)

def _parse_regular_expression1(m: Any, references:Dict[str,Any], factory: Any)->Any:
    if len(m.concatenations) > 0:
        expr = [ _parse_regular_expression_2(m.expression, references, factory) ]
        for n in m.concatenations:
            expr.append(_parse_regular_expression_2(n, references, factory))
        return factory['Concatenations'](expr)
    return _parse_regular_expression_2(m.expression, references, factory)

def _parse_regular_expression_2(m: Any, references:Dict[str,Any], factory: Any)->Any:
    if m.kleene:
        return factory['Kleene'](_parse_regular_expression_3(m.subexpression, references, factory))
    if m.omega:
        return factory['Omega'](_parse_regular_expression_3(m.subexpression, references, factory))
    else:
        return _parse_regular_expression_3(m.subexpression, references, factory)

def _parse_regular_expression_3(m: Any, references:Dict[str,Any], factory: Any)->Any:
    if m.emptyLangExpression:
        return factory['EmptyLanguage']()
    if m.emptyWordExpression:
        return factory['EmptyWord']()
    if m.letterExpression:
        return factory['Letter'](m.letterExpression)
    if m.referenceExpression:
        ref = m.referenceExpression.reference
        exp = references[ref]
        if not ref in references['_processed']:
            exp = _parse_regular_expression(exp, references, factory)
            references['_processed'].add(ref)
            references[ref] = exp
        return exp
    return _parse_regular_expression(m.expression, references, factory)
