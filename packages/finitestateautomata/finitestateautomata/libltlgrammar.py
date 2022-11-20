from typing import Any, Dict, List, Optional, Set, Tuple, Union
from textx import metamodel_from_str, TextXSyntaxError
import sys

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\LinearTemporalLogic.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


LTLGrammar = """

LTLModel:
	'ltl' 'formula' name=LTLID '='
	formula = LTLFormula	
	('alphabet' alphabet = SetOfSymbols 
	)?
	('where' (definitions = Definition )*
	)?
;

Definition:
	proposition = LTLID '=' symbols = SetOfSymbols
;

SetOfSymbols:
	'{' (symbols = LTLID) (',' symbols = LTLID)* '}'
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
		(subexpression1 = LTLFormula3) ('U' subexpression2 = LTLFormula2)?
;

LTLFormula3:
		(subexpression1 = LTLFormula4) ('R' subexpression2 = LTLFormula3)?
;

LTLFormula4:
		subexpression = LTLFormula5 ('=>' consequence = LTLFormula4)?
;

LTLFormula5:
		('X' nextSubexpression = LTLFormula5) | 
		('F' eventuallySubexpression = LTLFormula5) | 
		('G' alwaysSubexpression = LTLFormula5) |
		('not' notSubexpression = LTLFormula5) |
		subexpression = LTLFormula6
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
	LTLID | STRING
;

LTLID:
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

MetaModelLTL = metamodel_from_str(LTLGrammar, classes=[])

def parseLTLDSL(content: str, factory: Any)->Union[Tuple[None,None],Tuple[str,Tuple[Any,Optional[Set[str]],Optional[Dict[str,Set[str]]]]]]:
    try:
        model =  MetaModelLTL.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write("Syntax error in line %d col %d: %s" % (err.line, err.col, err.message))
        return (None, None)
    return (model.name, parseLTLModel(model, factory))

def parseLTLModel(model: Any, factory: Any)->Tuple[Any,Optional[Set[str]],Optional[Dict[str,Set[str]]]]:
    return (parseLTLFormula(model.formula, factory), parseAlphabet(model.alphabet), parseDefinitions(model.definitions))


def parseAlphabet(alphabet: Any)->Optional[Set[str]]:
    if alphabet is None:
        return None
    return set(alphabet.symbols)

def parseDefinitions(definitions)->Optional[Dict[str,Set[str]]]:
    if len(definitions) == 0:
        return None
    defs = dict()
    for d in definitions:
        defs[d.proposition] = set(d.symbols.symbols)
    return defs

def parseLTLFormula(f: Any, factory: Any)->Any:
    if len(f.alternatives) > 0:
        fs = [ parseLTLFormula1(f.formula, factory) ]
        for phi in f.alternatives:
            fs.append(parseLTLFormula1(phi, factory))
        return factory['Disjunction'](fs)
    else: 
        return parseLTLFormula1(f.formula, factory)

def parseLTLFormula1(f: Any, factory: Any)->Any:
    if len(f.alternatives) > 0:
        fs = [ parseLTLFormula2(f.formula, factory) ]
        for phi in f.alternatives:
            fs.append(parseLTLFormula2(phi, factory))
        return factory['Conjunction'](fs)
    else: 
        return parseLTLFormula2(f.formula, factory)

def parseLTLFormula2(f: Any, factory: Any)->Any:
    phi1 = parseLTLFormula3(f.subexpression1, factory)
    if f.subexpression2:
        phi2 = parseLTLFormula2(f.subexpression2, factory)

        return factory['Until'](phi1, phi2)
    else: 
        return phi1

def parseLTLFormula3(f: Any, factory: Any)->Any:
    phi1 = parseLTLFormula4(f.subexpression1, factory)
    if f.subexpression2:
        phi2 = parseLTLFormula3(f.subexpression2, factory)
        return factory['Release'](phi1, phi2)
    else: 
        return phi1

def parseLTLFormula4(f: Any, factory: Any)->Any:
    phi1 = parseLTLFormula5(f.subexpression, factory)
    if f.consequence:
        phi2 = parseLTLFormula4(f.consequence, factory)
        return factory['Disjunction']([factory['Negation'](phi1), phi2])
    else: 
        return phi1

def parseLTLFormula5(f: Any, factory: Any)->Any:
    if f.nextSubexpression:
        return factory['Next'](parseLTLFormula5(f.nextSubexpression, factory))
    if f.eventuallySubexpression:
        return factory['Eventually'](parseLTLFormula5(f.eventuallySubexpression, factory))
    if f.alwaysSubexpression:
        return factory['Always'](parseLTLFormula5(f.alwaysSubexpression, factory))
    if f.notSubexpression:
        return factory['Negation'](parseLTLFormula5(f.notSubexpression, factory))
    return parseLTLFormula6(f.subexpression, factory)

def parseLTLFormula6(f: Any, factory: Any)->Any:
    if f.trueExpression:
        return factory['True']()
    if f.falseExpression:
        return factory['False']()
    if f.propositionExpression:
        return factory['Proposition'](f.propositionExpression)
    else: 
        return parseLTLFormula(f.expression, factory)


