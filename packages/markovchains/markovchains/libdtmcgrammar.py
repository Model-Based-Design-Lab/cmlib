from textx import metamodel_from_str, TextXSyntaxError
import sys

DTMCGrammar = """
MarkovChainModel:
	('author' '=' author=ID)?
	'markov' 'chain' name=ID '{'
	edges += Edge*
	'}'
;

Edge:
        (srcstate=State ('-')+ '>' dststate=State) | (srcstate=State ('-')+ (specs=EdgeSpecs) ('-')+ '>' dststate=State)
;

EdgeSpecs:
	annotations += EdgeAnnotation (';' annotations += EdgeAnnotation)*
;

EdgeAnnotation:
	('probability' ':' probability=Probability)
	|
	(probability=Probability) 
;

State:
	name=ID
	(specs=StateSpecs)?
;

StateSpecs:
	'[' 
	annotations += StateAnnotation (';' annotations += StateAnnotation)*
	']'
;

StateAnnotation:
	(((('initial' 'probability') | ('p')) ':') initprob=Probability) |
	(((('reward') | ('r')) ':') reward=Reward) 	 
;

Reward:
	ratio=Ratio | float=Float | int=INT  
;

Probability:
	ratio=Ratio | float=Float | int=INT  
;

Ratio:
	numerator=INT '/' denominator=INT
;

Float: STRICTFLOAT;
"""

MetaModelDTMC = metamodel_from_str(DTMCGrammar, classes=[])

def parseDTMCDSL(content, factory):

	def _getProbabilityOrReward(p):
		if p.ratio != None:
			return float(p.ratio.numerator) / float(p.ratio.denominator)
		if p.float != None:
			return float(p.float)
		if p.int != None:
			return float(p.int)

	def _parseState(s):
		state = factory['AddState'](DTMC, s.name)
		if s.specs:
			for sa in s.specs.annotations:
				if sa.initprob:
					factory['SetInitialProbability'](DTMC, s.name, _getProbabilityOrReward(sa.initprob))
				if sa.reward:
					factory['SetReward'](DTMC, s.name, _getProbabilityOrReward(sa.reward))
		return state

	def _parseEdgeSpec(s, src, dst):
		if s.probability:
			factory['SetEdgeProbability'](DTMC, src, dst, _getProbabilityOrReward(s.probability))

	def _parseEdge(e):
		src = _parseState(e.srcstate)
		dst = _parseState(e.dststate)
		if e.specs:
			for a in e.specs.annotations:
				_parseEdgeSpec(a, src, dst)


	try:
		model =  MetaModelDTMC.model_from_str(content)
	except TextXSyntaxError as err:
		sys.stderr.write("Syntax error in line %d col %d: %s" % (err.line, err.col, err.message))
		return (None, None)

	DTMC = factory['Init']()

    # parse
	for e in model.edges:
		_parseEdge(e)

	factory['SortNames'](DTMC)

	DTMC.completeInitialProbabilities()
	DTMC.completeRewards()
	DTMC.addImplicitTransitions()

	return (model.name, DTMC)
    
