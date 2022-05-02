from textx import metamodel_from_str, TextXSyntaxError
import sys

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\RegularExpressions.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


RegExGrammar = """
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
	SIMPLELETTER | STRING
;
SIMPLELETTER:
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

MetaModelRegEx = metamodel_from_str(RegExGrammar, classes=[])

def parseRegExDSL(content, factory):
    try:
        model =  MetaModelRegEx.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write("Syntax error in line %d col %d: %s" % (err.line, err.col, err.message))
        return (None, None)
    regex = parseRefsAndRegularExpression(model, factory)
    return (model.name, regex)

def parseRefsAndRegularExpression(m, factory):
    references = dict()
    if m.definitions:
        references = parseRefs(m.definitions, factory)
    return parseRegularExpression(m.expression, references, factory)

def parseRefs(defs, factory):
    res = dict()
    res['_processed'] = set()
    for d in defs:
        res[d.symbol] = d.expression
    return res    

def parseRegularExpression(m, references, factory):
    if len(m.alternatives) > 0:
        expr = [ parseRegularExpression1(m.expression, references, factory) ]
        for n in m.alternatives:
            expr.append(parseRegularExpression1(n, references, factory))
        return factory['Alternatives'](expr)
    else: 
        return parseRegularExpression1(m.expression, references, factory)

def parseRegularExpression1(m, references, factory):
    if len(m.concatenations) > 0:
        expr = [ parseRegularExpression2(m.expression, references, factory) ]
        for n in m.concatenations:
            expr.append(parseRegularExpression2(n, references, factory))
        return factory['Concatenations'](expr)
    else: 
        return parseRegularExpression2(m.expression, references, factory)

def parseRegularExpression2(m, references, factory):
    if m.kleene:
        return factory['Kleene'](parseRegularExpression3(m.subexpression, references, factory))
    if m.omega:
        return factory['Omega'](parseRegularExpression3(m.subexpression, references, factory))
    else: 
        return parseRegularExpression3(m.subexpression, references, factory)

def parseRegularExpression3(m, references, factory):
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
            exp = parseRegularExpression(exp, references, factory)
            references['_processed'].add(ref)
            references[ref] = exp
        return exp
    return parseRegularExpression(m.expression, references, factory)
