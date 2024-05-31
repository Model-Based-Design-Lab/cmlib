from fractions import Fraction
from typing import Any, Optional, Tuple
from textx import metamodel_from_str, TextXSyntaxError
import sys

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\MarkovChains.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


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

def parseDTMCDSL(content: str, factory: Any)->Tuple[Optional[str],Optional[Any]]:

	def _getProbabilityOrReward(p: Any)->Optional[Fraction]:
		'''Parse probability.'''
		if p.ratio != None:
			return Fraction(int(p.ratio.numerator), int(p.ratio.denominator))
		if p.float != None:
			return Fraction(p.float)
		if p.int != None:
			return Fraction(p.int)

	def _parseState(s: Any)->Any:
		'''Parse state with attributes.'''
		state = factory['AddState'](DTMCModel, s.name)
		if s.specs:
			for sa in s.specs.annotations:
				if sa.initprob:
					factory['SetInitialProbability'](DTMCModel, s.name, _getProbabilityOrReward(sa.initprob))
				if sa.reward:
					factory['SetReward'](DTMCModel, s.name, _getProbabilityOrReward(sa.reward))
		return state

	def _parseEdgeSpec(s: Any, src: str, dst: str):
		'''Parse transition attributes.'''
		if s.probability:
			factory['SetEdgeProbability'](DTMCModel, src, dst, _getProbabilityOrReward(s.probability))

	def _parseEdge(e: Any):
		'''Parse a transition.'''
		src: str = _parseState(e.srcstate)
		dst: str = _parseState(e.dststate)
		if e.specs:
			for a in e.specs.annotations:
				_parseEdgeSpec(a, src, dst)


	try:
		model: Any =  MetaModelDTMC.model_from_str(content)
	except TextXSyntaxError as err:
		sys.stderr.write("Syntax error in line %d col %d: %s" % (err.line, err.col, err.message))
		return (None, None)

	DTMCModel: Any = factory['Init']()

    # parse
	for e in model.edges:
		_parseEdge(e)

	factory['SortNames'](DTMCModel)

	DTMCModel.complete_initial_probabilities()
	DTMCModel.complete_rewards()
	DTMCModel.add_implicit_transitions()

	return (model.name, DTMCModel)

