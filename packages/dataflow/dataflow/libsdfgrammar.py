from typing import Union,Tuple,Any
from textx import metamodel_from_str, TextXSyntaxError
import sys

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\Dataflow.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git

SDFGrammar = """

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
	STRICTFLOAT | '-'? INT
;


Comment:
    /\\/\\*(.|\\n)*?\\*\\// | /\\/\\/.*?$/
;

"""

MetaModelSDF = metamodel_from_str(SDFGrammar, classes=[])

def parseSDFDSL(content, factory)->Union[Tuple[None,None],Tuple[str,Any]]:
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
    try:
        model =  MetaModelSDF.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write("Syntax error in line %d col %d: %s" % (err.line, err.col, err.message))
        return (None, None)
    sdf = factory['Init']()
    for e in model.edges:
        parseEdge(e, sdf, factory)
    if model.inputs:
        for i in model.inputs.ports:
            factory['AddInputPort'](sdf, i.name)
    if model.outputs:
        for o in model.outputs.ports:
            factory['AddOutputPort'](sdf, o.name)

    if model.inputsignals:
        for insig in model.inputsignals.signals:
            factory['AddInputSignal'](sdf, insig.name, [float(ts) for ts in insig.timestamps])

    return (model.name, sdf)

def parseActorSpecs(specs):
	res = dict()
	for a in specs.annotations:
		if isinstance(a.executiontime, str):
			res['executionTime'] = float(a.executiontime)
		else:
			res['executionTime'] = a.executiontime
	return res


def parseActor(a, sdf, factory):
	specs = dict()
	if a.specs is not None:
		specs = parseActorSpecs(a.specs)
	factory['AddActor'](sdf, a.name, specs)
	return a.name


def parseEdgeSpecs(specs):
	res = dict()
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

def parseEdge(e, sdf, factory):
    srcActor = parseActor(e.srcact, sdf, factory)
    dstActor = parseActor(e.dstact, sdf, factory)

    specs = dict()
    if e.specs:
        specs = parseEdgeSpecs(e.specs)
    factory['AddChannel'](sdf, srcActor, dstActor, specs)
    
