'''DSL grammar parsing support for max-plus models.'''

import sys
from fractions import Fraction
from typing import Any, Dict, Tuple, Union

from textx import TextXSyntaxError, metamodel_from_str

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\MaxPlusMatrix.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git

MIN_INF_GRAMMAR_STRING = "-inf"

MPM_GRAMMAR = """

MaxPlusMatrixModel:
	'max-plus' 'model' name=ID ':'
	('matrices' (matrices = MaxPlusMatrix)+)?
	('vector sequences' (vectorsequences = VectorSequence)+)?
	('event sequences' (eventsequences = EventSequence)+)?
	;

MaxPlusMatrix:
	name=ID
	(labels = Labels  (',')?)?
	'='
	'['
	(rows = Row (',')?)*
	']'
;

VectorSequence:
	name=ID
	(labels = Labels (',')?)?
	'='
	'['
	(vectors += Row (',')?)*
	']'
;

EventSequence:
	name=ID '='
	sequence = Row
;

Labels:
	'('
	(label = ID (',')? )+
	')'
;

Row:
	'[' ( elements = Element (',')?)* ']'
;

Element:
	Number | '-inf'
;


Number:
	ratio=Ratio | float=Float | int=Int
;

Ratio:
	numerator=Int '/' denominator=INT
;

Float: STRICTFLOAT;

Int: '-'? INT;

Comment:
    /\\/\\*(.|\\n)*?\\*\\// | /\\/\\/.*?$/
;

"""

MetaModelMPM = metamodel_from_str(MPM_GRAMMAR, classes=[])

class MPMParsingException(Exception):
    '''Parsing exception.'''

def parse_mpm_dsl(content: str, factory: Dict[str,Any]) -> Union[Tuple[None,None,None,None], \
                            Tuple[str,Dict[str,Any],Dict[str,Any],Dict[str,Any]]]:
    '''Parse max-plus model from DSL string.'''

    def _get_number(n):
        if n.ratio is not None:
            return Fraction(f"{n.ratio.numerator}/{n.ratio.denominator}").limit_denominator()
        if n.float is not None:
            return Fraction(n.float).limit_denominator()
        if n.int is not None:
            return Fraction(n.int).limit_denominator()
        # we cannot get here
        raise MPMParsingException("Parser error.")

    def _parse_row(r: Any, mpm: Any, factory: Dict[str,Any]):
        row = []
        for e in r.elements:
            if e == MIN_INF_GRAMMAR_STRING:
                row.append(None)
            else:
                row.append(_get_number(e))

        factory['AddRow'](mpm, row)

    def _parse_vector(v: Any, mpm: Any, factory: Dict[str,Any]):
        vc = []
        for e in v.elements:
            if e == MIN_INF_GRAMMAR_STRING:
                vc.append(None)
            else:
                vc.append(_get_number(e))

        factory['AddVector'](mpm, vc)

    def _set_event_sequence(es: Any, mpm: Any, factory: Dict[str,Any]):
        seq = []
        for e in es.elements:
            if e == MIN_INF_GRAMMAR_STRING:
                seq.append(None)
            else:
                seq.append(_get_number(e))

        factory['SetSequence'](mpm, seq)

    def _parse_labels(labels):
        return labels.label

    try:
        model =  MetaModelMPM.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write(f"Syntax error in line {err.line} col {err.col}: {err.message}\n")
        return (None, None, None, None)

    res_matrices = {}
    for m in model.matrices:
        mpm = factory['Init']()
        if m.labels:
            factory['AddLabels'](mpm, _parse_labels(m.labels))
        for vc in m.rows:
            _parse_row(vc, mpm, factory)
        res_matrices[m.name] = mpm

    res_vector_sequences = {}
    for v in model.vectorsequences:
        vs = factory['InitVectorSequence']()
        if v.labels:
            factory['AddLabels'](vs, _parse_labels(v.labels))
        for vc in v.vectors:
            _parse_vector(vc, vs, factory)
        res_vector_sequences[v.name] = vs

    res_event_sequences = {}
    for e in model.eventsequences:
        es = factory['InitEventSequence']()
        _set_event_sequence(e.sequence, es, factory)
        res_event_sequences[e.name] = es

    return model.name, res_matrices, res_vector_sequences, res_event_sequences
