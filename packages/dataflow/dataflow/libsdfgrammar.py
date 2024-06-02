'''DSL grammar parsing support for max-plus models.'''

import sys
from fractions import Fraction
from typing import Any, Tuple, Union

from textx import TextXSyntaxError, metamodel_from_str

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\Dataflow.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git

MIN_INF_GRAMMAR_STRING = "-inf"
MIN_INF_VAL = None

SDF_GRAMMAR = """

DataflowModel:
	('author' '=' author=ID)?
	'dataflow' 'graph' name=ID '{'
	('inputs' inputs=PortList)?
	('outputs' outputs=PortList)?
	edges += Edge*
	'}'
	(inputsignals = InputSignals)?
;

Edge:
		srcact=Actor
        ('-')+
		(( '>' dstact=Actor | ( (specs=EdgeSpecs) ('-')+ '>' dstact=Actor)))
;

EdgeSpecs:
	annotations += EdgeAnnotation (';' annotations += EdgeAnnotation)*
;

EdgeAnnotation:
	(('initial' 'tokens' ':' )? initialtokens=INT)
	|
	('production' 'rate' ':' prodrate=INT)
	|
	('consumption' 'rate' ':' consrate=INT)
	|
	('name' ':' name=ID)
	|
	('token' 'size' ':' tokensize=INT)
;

Actor:
	name=ID
	(specs=ActorSpecs)?
;

ActorSpecs:
	'['
	annotations += ActorAnnotation (';' annotations += ActorAnnotation)*
	']'
;

ActorAnnotation:
	( ('execution' 'time' ':')? executiontime=Number)
;

PortList:
	(ports = Port) (',' ports = Port)*
;

Port:
	name=ID
;


InputSignals:
	'input' 'signals'
	 signals+=Signal
;

Signal:
	name=ID '='
	'['
	 (
		 ( timestamps = TimeStamp)
	 	(',' timestamps = TimeStamp)*
	)?
	']'
;

TimeStamp:
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

MetaModelSDF = metamodel_from_str(SDF_GRAMMAR, classes=[])

class SDFParsingException(Exception):
    '''Parsing exception.'''


def parse_sdf_dsl(content, factory)->Union[Tuple[None,None],Tuple[str,Any]]:
    '''
    Parse the provided content. Using the factory operations to build the result.
    Returns a pair with the name of the model and the constructed result.
    factory is a dictionary with lambda functions with the following keys:
    - 'Init' = lambda
    - 'AddActor' = lambda sdf, a, specs
    - 'AddChannel' = lambda sdf, a1, a2, specs
    - 'AddInputPort' = lambda sdf, i
    - 'AddOutputPort' = lambda sdf, i
    - 'AddInputSignal' = lambda sdf, n, s
    '''

    def _get_number(n):
        if n.ratio is not None:
            return Fraction(f"{n.ratio.numerator}/{n.ratio.denominator}").limit_denominator()
        if n.float is not None:
            return Fraction(n.float).limit_denominator()
        if n.int is not None:
            return Fraction(n.int).limit_denominator()
        # we cannot get here
        raise SDFParsingException("Parser error.")

    def _parse_actor_specs(specs):
        res = {}
        for a in specs.annotations:
            res['executionTime'] = _get_number(a.executiontime)
        return res


    def _parse_actor(a, sdf, factory):
        specs = {}
        if a.specs is not None:
            specs = _parse_actor_specs(a.specs)
        factory['AddActor'](sdf, a.name, specs)
        return a.name


    def _parse_edge_specs(specs):
        res = {}
        for a in specs.annotations:
            if a.initialtokens != 0:
                res['initialTokens'] = a.initialtokens
            if a.prodrate != 0:
                res['prodRate'] = a.prodrate
            if a.consrate != 0:
                res['consRate'] = a.consrate
            if a.name != '':
                res['name'] = a.name
            if a.tokensize != 0:
                res['tokenSize'] = a.tokensize
        return res

    def _parse_edge(e, sdf, factory):
        src_actor = _parse_actor(e.srcact, sdf, factory)
        dst_actor = _parse_actor(e.dstact, sdf, factory)

        specs = {}
        if e.specs:
            specs = _parse_edge_specs(e.specs)
        factory['AddChannel'](sdf, src_actor, dst_actor, specs)

    try:
        model =  MetaModelSDF.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write(f"Syntax error in line {err.line} col {err.col}: {err.message}")
        return (None, None)
    sdf = factory['Init']()
    for e in model.edges:
        _parse_edge(e, sdf, factory)
    if model.inputs:
        for i in model.inputs.ports:
            factory['AddInputPort'](sdf, i.name)
    if model.outputs:
        for o in model.outputs.ports:
            factory['AddOutputPort'](sdf, o.name)

    if model.inputsignals:
        for in_sig in model.inputsignals.signals:
            factory['AddInputSignal'](sdf, in_sig.name, [_get_number(ts) if \
                    ts!=MIN_INF_GRAMMAR_STRING else MIN_INF_VAL for ts in in_sig.timestamps])

    return (model.name, sdf)
