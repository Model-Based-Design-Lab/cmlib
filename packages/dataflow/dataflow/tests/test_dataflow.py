from fractions import Fraction
import os
import pytest
import copy
from modeltest.modeltest import Model_pytest
from dataflow.libsdf import DataflowGraph
from dataflow.libmpm import MaxPlusMatrixModel
from dataflow.utils.utils import parseInitialState, getSquareMatrix
from dataflow.utils.commandline import _convolution, _maximum

TEST_FILE_FOLDER = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(TEST_FILE_FOLDER, "models")
OUTPUT_FOLDER = os.path.join(TEST_FILE_FOLDER, "output")
MODEL_SDF_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".sdf")]
MODEL_MPM_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".mpm")]



###############################################
### Synchronous Data Flow Models            ###
###############################################

class SDF_pytest(Model_pytest):
    def __init__(self, model):
        
        # First load LTL model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".sdf")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")
                
        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r') as sdfFile:
            dsl = sdfFile.read()
        self.name, self.model = DataflowGraph.fromDSL(dsl)

        # Create namespace of user arguments
        self.args = self.Namespace(
            initialstate=None
        )

        # Safe deadlock status
        self.mu = None
        self.deadlock = self.model.deadlock()
        if not self.deadlock:
            self.throughput = self.model.throughput()
            if self.throughput != "infinite":
                # Derive smallest period
                if not self.deadlock:
                    self.mu = 1/self.throughput


    def Correct_behavior_tests(self):
        self.function_test(lambda: self.model.deadlock(), "deadlock")
        self.function_test(lambda: self.model.repetitionVector(), "repetitionVector")
        self.function_test(lambda: self.model.listOfInputsStr(), "listOfInputsStr")
        self.function_test(lambda: self.model.listOfOutputsStr(), "listOfOutputsStr")
        self.function_test(lambda: self.model.listOfStateElementsStr(), "listOfStateElementsStr")
        inv_model = copy.deepcopy(self.model) # Need copy of current object for save conversion
        self.function_test(lambda: vars(inv_model.convertToSingleRate()), "convertToSingleRate", sort = True)
        if not self.deadlock:
            self.function_test(lambda: self.model.throughput(), "throughput")
            self.function_test(lambda: self.model.stateSpaceMatrices(), "stateSpaceMatrices")
        mu = self.mu
        if mu is not None:
            x0 = parseInitialState(self.args, self.model.numberOfInitialTokens())
            self.function_test(lambda: self.model.latency(x0, mu), "latency")
            self.function_test(lambda: self.model.generalizedLatency(mu), "generalizedLatency")

    def Incorrect_behavior_tests(self):
        mu = self.mu
        if mu is not None:
            x0 = parseInitialState(self.args, self.model.numberOfInitialTokens())
            self.incorrect_test(lambda: self.model.latency(x0, mu-Fraction(0.01)), "The request period mu is smaller than smallest period the system can sustain. Therefore, it has no latency.")
            self.incorrect_test(lambda: self.model.generalizedLatency(mu-Fraction(0.01)), "The request period mu is smaller than smallest period the system can sustain. Therefore, it has no latency.")

    # Class to create namespace args consisting of user input arguments 
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


@pytest.mark.parametrize("test_model", MODEL_SDF_FILES)
def test_SDF(test_model: str):
    m = SDF_pytest(test_model)
    m.Correct_behavior_tests()
    m.Incorrect_behavior_tests()
    m.write_output_file()


###############################################
### Max-Plus Models                         ###
###############################################

class MPM_pytest(Model_pytest):
    def __init__(self, model):
        
        # First load LTL model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".mpm")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")
                
        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r') as mpmFile:
            dsl = mpmFile.read()
        self.name, self.Matrices, self.VectorSequences, self.EventSequences = MaxPlusMatrixModel.fromDSL(dsl)

        sq = ",".join(self.EventSequences.keys())
        sq = ",".join([sq, sq])

        # Create namespace of user arguments
        self.args = self.Namespace(
            initialstate=None,
            inputtrace=None,
            matrices=None,
            numberofiterations=10,
            outputGraph=None,
            parameter=None,
            period=None,
            sequences=sq,
        )

    def Correct_behavior_tests(self):
        # Function used to obtain multiple results
        mat = getSquareMatrix(self.Matrices, self.args)

        # Deterministic results
        self.function_test(lambda: self.Matrices[mat].eigenvalue(), "Eigenvalues")
        self.function_test(lambda: self.Matrices[mat].starClosure(), "starClosure")
        
        # Only check convolution when sequence is available
        if len(self.args.sequences) > 1:  # type: ignore
            sequences, res = _convolution(self.EventSequences, self.args)
            self.function_test(lambda: sequences, "convolution_sequences")
            self.function_test(lambda: vars(res)['_sequence'], "convolution_res")
            res = self.EventSequences[list(self.EventSequences.keys())[0]].delay(self.args.numberofiterations)  # type: ignore
            self.function_test(lambda: vars(res)['_sequence'], "delay_sequence")
            res = self.EventSequences[list(self.EventSequences.keys())[0]].scale(self.args.numberofiterations)  # type: ignore
            self.function_test(lambda: vars(res)['_sequence'], "scale_sequence")
            sequences, res = _maximum(self.EventSequences, self.args)
            self.function_test(lambda: sequences, "maximum_analysis_sequences")
            self.function_test(lambda: vars(res)['_sequence'], "maximum_analysis_res")
        success, cl = self.Matrices[list(self.Matrices.keys())[0]].starClosure()
        # if success:
        self.function_test(lambda: cl, "star_closure")
        matrix = self.Matrices[list(self.Matrices.keys())[0]]
        self.function_test(lambda: vars(MaxPlusMatrixModel.multiplySequence([matrix]))["_rows"], "multiply")

        # Non deterministic results, False added as argument
        self.function_test(lambda: self.Matrices[mat].eigenvectors(), "Eigenvectors", False) # Eigenvectors can appear in different order

    def Incorrect_behavior_tests(self):
        pass

    # Class to create namespace args consisting of user input arguments 
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

@pytest.mark.parametrize("test_model", MODEL_MPM_FILES)
def test_MPM(test_model: str):
    m = MPM_pytest(test_model)
    m.Correct_behavior_tests()
    m.Incorrect_behavior_tests()
    m.write_output_file()