'''DSL grammar parsing support for Markov Chains.'''
import sys
from fractions import Fraction
from typing import Any, Optional, Tuple

from textx import TextXSyntaxError, metamodel_from_str

# This is TextX version of the XText grammar in the clang repository
# Xtext file:
# src\info.computationalmodeling.lang.parent\info.computationalmodeling.lang\src\main\java\info\computationalmodeling\lang\MarkovChains.xtext
# repository:
# https://git.ics.ele.tue.nl/computational-modeling/cmlang.git


DTMC_GRAMMAR = """
MarkovChainModel:
    ('author' '=' author=ID)?
    'markov' 'chain' name=ID '{'
    edges += Edge*
    '}'
;

Edge:
        (src_state=State ('-')+ '>' dst_state=State) | (src_state=State ('-')+ (specs=EdgeSpecs) ('-')+ '>' dst_state=State)
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
    (((('initial' 'probability') | ('p')) ':') init_prob=Probability) |
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

MetaModelDTMC = metamodel_from_str(DTMC_GRAMMAR, classes=[])

def parse_dtmc_dsl(content: str, factory: Any)->Tuple[Optional[str],Optional[Any]]:
    '''Parse a Markov Chain from DSL string.'''

    def _get_probability_or_reward(p: Any)->Optional[Fraction]:
        '''Parse probability.'''
        if p.ratio is not None:
            return Fraction(int(p.ratio.numerator), int(p.ratio.denominator))
        if p.float is not None:
            return Fraction(p.float)
        if p.int is not None:
            return Fraction(p.int)

    def _parse_state(s: Any)->Any:
        '''Parse state with attributes.'''
        state = factory['AddState'](dtmc_model, s.name)
        if s.specs:
            for sa in s.specs.annotations:
                if sa.init_prob:
                    factory['SetInitialProbability'](dtmc_model, \
								s.name, _get_probability_or_reward(sa.init_prob))
                if sa.reward:
                    factory['SetReward'](dtmc_model, s.name, _get_probability_or_reward(sa.reward))
        return state

    def _parse_edge_spec(s: Any, src: str, dst: str):
        '''Parse transition attributes.'''
        if s.probability:
            factory['SetEdgeProbability'](dtmc_model, src, dst, \
						_get_probability_or_reward(s.probability))

    def _parse_edge(e: Any):
        '''Parse a transition.'''
        src: str = _parse_state(e.src_state)
        dst: str = _parse_state(e.dst_state)
        if e.specs:
            for a in e.specs.annotations:
                _parse_edge_spec(a, src, dst)


    try:
        model: Any =  MetaModelDTMC.model_from_str(content)
    except TextXSyntaxError as err:
        sys.stderr.write(f"Syntax error in line {err.line} col {err.col}: {err.message}")
        return (None, None)

    dtmc_model: Any = factory['Init']()

    # parse
    for e in model.edges:
        _parse_edge(e)

    factory['SortNames'](dtmc_model)

    dtmc_model.complete_initial_probabilities()
    dtmc_model.complete_rewards()
    dtmc_model.add_implicit_transitions()

    return (model.name, dtmc_model)
