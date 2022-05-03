import os
import pytest
from modeltest.modeltest import Model_pytest
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


###############################################
### Finite state automata model tests       ###
###############################################

class Automaton_pytest(Model_pytest):
    def __init__(self, model):

        # First load finitestateautomata model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".fsa")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")
                
        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r') as fsaFile:
            dsl = fsaFile.read()
        self.name, self.model = Automaton.fromDSL(dsl)

    def Correct_behaviour_tests(self):
        self.function_test(lambda: self.model.alphabet(), "alphabet", sort = True, quotes = True)
        self.function_test(lambda: self.model.states(), "states", sort = True)
        self.function_test(lambda: self.model.isDeterministic(), "deterministic")
        self.function_test(lambda: self.model.reachableStates(), "reachableStates", sort = True)
        self.function_test(lambda: self.model.languageEmpty(), "languageEmpty")
        self.function_test(lambda: self.model.accepts('h,s,m,d,p'), "accepts") # Arbitrary word
        self.function_test(lambda: self.model.accepts(''), "accepts_empty") # Arbitrary word
        self.function_test(lambda: self.model.languageIncluded(self.model), "languageIncluded")
        self.function_test(lambda: vars(self.model.product(self.model)), "product", sort = True)
        self.function_test(lambda: vars(self.model.eliminateEpsilonTransitions()), "eliminateEpsilon", sort = True)
        self.function_test(lambda: vars(self.model.complete()), "complete", sort = True)
        self.function_test(lambda: vars(self.model.complement()), "complement", sort = True)
        self.function_test(lambda: vars(self.model.minimize()), "minimize", sort = True)
        self.function_test(lambda: vars(self.model.asDFA()), "convert_to_DFA", sort = True)
        self.function_test(lambda: vars(self.model.relabelStates()), "relabel", sort = True) # Nondeterministic

        # Test current model against reference model defined in /models folder 
        if "fsa_refModel.fsa" in MODEL_FSA_FILES:
            loc = os.path.join(MODEL_FOLDER, "fsa_refModel.fsa")
            with open(loc, 'r') as secondModel:
                dsl = secondModel.read()
            name, model = Automaton.fromDSL(dsl)
            self.function_test(lambda: self.model.languageIncluded(model), "languageIncluded_refModel")
            self.function_test(lambda: vars(self.model.product(model)), "product_refModel", sort = True)

        self.function_test(lambda: self.model.asRegularBuchiAutomaton().languageEmptyBuchiAlternative(), "languageEmptyBuchi", sort = True)
        self.function_test(lambda: vars(self.model.asRegularBuchiAutomaton().productBuchi(self.model.asRegularBuchiAutomaton())), "productBuchi", sort = True)
        self.function_test(lambda: vars(self.model.minimizeBuchi()), "minimizeBuchi", sort = True)

    def Incorrect_behaviour_tests(self):
        pass

@pytest.mark.parametrize("test_model", MODEL_FSA_FILES)
def test_automaton(test_model: str):
    m = Automaton_pytest(test_model)
    m.Correct_behaviour_tests()
    m.Incorrect_behaviour_tests()
    m.write_output_file()




###############################################
### Linear-time Temporal Logic model tests  ###
###############################################

class LTL_pytest(Model_pytest):
    def __init__(self, model):
        
        # First load LTL model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".ltl")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")
                
        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r') as ltlFile:
            dsl = ltlFile.read()
        self.name, self.model = LTLFormula.fromDSL(dsl)

    def Correct_behaviour_tests(self):
        self.function_test(lambda: vars(self.model.asFSA()), "convert_to_NBA", sort = True)

    def Incorrect_behaviour_tests(self):
        pass

@pytest.mark.parametrize("test_model", MODEL_LTL_FILES)
def test_LTL(test_model: str):
    m = LTL_pytest(test_model)
    m.Correct_behaviour_tests()
    m.Incorrect_behaviour_tests()
    m.write_output_file()


###############################################
### Regular expression model tests  ###
###############################################

class Regular_expression_pytest(Model_pytest):
    def __init__(self, model):
        
        # First load regular expression
        self.model_name = model[:-6]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".regex")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")
                
        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r') as regFile:
            dsl = regFile.read()
        self.name, self.model = RegEx.fromDSL(dsl)

    def Correct_behaviour_tests(self):
        if self.model.isFiniteRegEx():
            self.function_test(lambda: vars(self.model.asFSA()), "convert_to_FSA", sort = True)
        
        if self.model.isOmegaRegEx():
            self.function_test(lambda: vars(self.model.asNBA()), "convert_to_NBA", sort = True)

    def Incorrect_behaviour_tests(self):
        pass


@pytest.mark.parametrize("test_model", MODEL_REGEX_FILES)
def test_regular_expression(test_model: str):
    m = Regular_expression_pytest(test_model)
    m.Correct_behaviour_tests()
    m.Incorrect_behaviour_tests()
    m.write_output_file()