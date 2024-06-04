'''
###############################################
### Finite state automata model tests       ###
###############################################
'''
import os
import pytest
from modeltest.modeltest import ModelPytest
from finitestateautomata.libfsa import Automaton
from finitestateautomata.libregex import RegEx
from finitestateautomata.libltl import LTLFormula

# Collect models and output files
TEST_FILE_FOLDER = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(TEST_FILE_FOLDER, "models")
OUTPUT_FOLDER = os.path.join(TEST_FILE_FOLDER, "output")
MODEL_FSA_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".fsa")]
MODEL_LTL_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".ltl")]
MODEL_REGEX_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".regex")]

class AutomatonPytest(ModelPytest):
    '''Testing finite state automata.'''
    def __init__(self, model):

        # First load finitestateautomata model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".fsa")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")

        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r', encoding="utf-8") as fsa_file:
            dsl = fsa_file.read()
        self.name, self.model = Automaton.from_dsl(dsl)

    def correct_behavior_tests(self):
        '''Testing behaviors that are expected to be successful.'''
        self.function_test(self.model.alphabet, "alphabet", sort = True, quotes = True)
        self.function_test(self.model.states, "states", sort = True)
        self.function_test(self.model.is_deterministic, "deterministic")
        self.function_test(self.model.reachable_states, "reachableStates", sort = True)
        if not self.model.has_generalized_acceptance_sets():
            self.function_test(self.model.language_empty, "languageEmpty")
        self.function_test(lambda: self.model.accepts('h,s,m,d,p'), "accepts") # Arbitrary word
        self.function_test(lambda: self.model.accepts(''), "accepts_empty") # Arbitrary word
        self.function_test(lambda: self.model.language_included(self.model), "languageIncluded")
        self.function_test(lambda: vars(self.model.product(self.model)), "product", sort = True)
        self.function_test(lambda: vars(self.model.eliminate_epsilon_transitions()), \
                            "eliminateEpsilon", sort = True)
        self.function_test(lambda: vars(self.model.complete()), "complete", sort = True)
        self.function_test(lambda: vars(self.model.complement()), "complement", sort = True)
        self.function_test(lambda: vars(self.model.minimize()), "minimize", sort = True)
        self.function_test(lambda: vars(self.model.as_dfa()), "convert_to_DFA", sort = True)
        self.function_test(lambda: vars(self.model.relabel_states()), "relabel", \
                           sort = True) # Nondeterministic
        self.function_test(lambda: self.model.as_dsl("TestName"), "convert_to_DSL", sort = True)

        # Test current model against reference model defined in /models folder
        if "fsa_refModel.fsa" in MODEL_FSA_FILES:
            loc = os.path.join(MODEL_FOLDER, "fsa_refModel.fsa")
            with open(loc, 'r', encoding='utf-8') as second_model:
                dsl = second_model.read()
            _, model = Automaton.from_dsl(dsl)
            self.function_test(lambda: self.model.language_included(model), \
                               "languageIncluded_refModel")
            self.function_test(lambda: vars(self.model.product(model)), "product_refModel", \
                               sort = True)

        self.function_test(lambda: self.model.as_regular_buchi_automaton().language_empty_buchi(), \
                           "languageEmptyBuchi", sort = True)
        self.function_test(lambda: vars(self.model.as_regular_buchi_automaton().product_buchi(\
            self.model.as_regular_buchi_automaton())), "productBuchi", sort = True)
        self.function_test(lambda: vars(self.model.minimize_buchi()), "minimizeBuchi", sort = True)

    def incorrect_behavior_tests(self):
        '''Testing behaviors that are expected to fail.'''

@pytest.mark.parametrize("model_under_test", MODEL_FSA_FILES)
def test_automaton(model_under_test: str):
    '''Test an automaton.'''
    m = AutomatonPytest(model_under_test)
    m.correct_behavior_tests()
    m.incorrect_behavior_tests()
    m.write_output_file()




###############################################
### Linear-time Temporal Logic model tests  ###
###############################################

class LTLPytest(ModelPytest):
    '''Testing linear temporal logic.'''
    def __init__(self, model):

        # First load LTL model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".ltl")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")

        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r', encoding='utf-8') as ltl_file:
            dsl = ltl_file.read()
        self.name, self.model = LTLFormula.from_dsl(dsl)

    def correct_behavior_tests(self):
        '''Testing behaviors that are expected to be successful.'''
        self.function_test(lambda: vars(self.model.as_fsa()), "convert_to_NBA", sort = True)
        self.function_test(lambda: self.model.as_dsl("TestName"), "convert_to_DSL", sort = True)

    def incorrect_behavior_tests(self):
        '''Testing behaviors that are expected to fail.'''

@pytest.mark.parametrize("model_under_test", MODEL_LTL_FILES)
def test_ltl(model_under_test: str):
    '''Test linear temporal logic formulas.'''
    m = LTLPytest(model_under_test)
    m.correct_behavior_tests()
    m.incorrect_behavior_tests()
    m.write_output_file()


###############################################
### Regular expression model tests  ###
###############################################

class RegularExpressionPytest(ModelPytest):
    '''Testing regular expressions.'''
    def __init__(self, model):

        # First load regular expression
        self.model_name = model[:-6]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".regex")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")

        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r', encoding='utf-8') as reg_file:
            dsl = reg_file.read()
        self.name, self.model = RegEx.from_dsl(dsl)

    def correct_behavior_tests(self):
        '''Testing behaviors that are expected to be successful.'''
        if self.model.is_finite_reg_ex():
            self.function_test(lambda: vars(self.model.as_fsa()), "convert_to_FSA", sort = True)

        if self.model.is_omega_reg_ex():
            self.function_test(lambda: vars(self.model.as_nba()), "convert_to_NBA", sort = True)
        self.function_test(lambda: self.model.as_dsl("TestName"), "convert_to_DSL", sort = True)

    def incorrect_behavior_tests(self):
        '''Testing behaviors that are expected to fail.'''


@pytest.mark.parametrize("test_model", MODEL_REGEX_FILES)
def test_regular_expression(test_model: str):
    '''Test a regular expression.'''
    m = RegularExpressionPytest(test_model)
    m.correct_behavior_tests()
    m.incorrect_behavior_tests()
    m.write_output_file()
