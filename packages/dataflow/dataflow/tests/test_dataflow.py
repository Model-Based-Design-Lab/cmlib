"""Testing the dataflow library"""

import copy
import os

from fractions import Fraction

import pytest
from dataflow.libmpm import MaxPlusMatrixModel
from dataflow.libsdf import DataflowGraph
from dataflow.utils.commandline import _convolution, _maximum
from dataflow.utils.utils import get_square_matrix, parse_initial_state
from modeltest.modeltest import ModelPytest

TEST_FILE_FOLDER = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(TEST_FILE_FOLDER, "models")
OUTPUT_FOLDER = os.path.join(TEST_FILE_FOLDER, "output")
MODEL_SDF_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".sdf")]
MODEL_MPM_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".mpm")]



###############################################
### Synchronous Data Flow Models            ###
###############################################

class SDFPyTest(ModelPytest):
    """Testing the dataflow domain."""
    def __init__(self, model):

        # First load LTL model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".sdf")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")

        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r', encoding='utf-8') as sdf_file:
            dsl = sdf_file.read()
        self.name, self.model = DataflowGraph.from_dsl(dsl)

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


    def correct_behavior_tests(self):
        """Testing behaviors with errors."""
        self.function_test(self.model.deadlock, "deadlock")
        self.function_test(self.model.repetition_vector, "repetitionVector")
        self.function_test(self.model.list_of_inputs_str, "listOfInputsStr")
        self.function_test(self.model.list_of_outputs_str, "listOfOutputsStr")
        self.function_test(self.model.list_of_state_elements_str, "listOfStateElementsStr")
        inv_model = copy.deepcopy(self.model) # Need copy of current object for save conversion
        self.function_test(lambda: vars(inv_model.convert_to_single_rate()),  \
                           "convertToSingleRate", sort = True)
        if not self.deadlock:
            self.function_test(self.model.throughput, "throughput")
            self.function_test(self.model.state_space_matrices, "stateSpaceMatrices")
        mu = self.mu
        if mu is not None:
            x0 = parse_initial_state(self.args, self.model.number_of_initial_tokens())
            self.function_test(lambda: self.model.latency(x0, mu), "latency")
            self.function_test(lambda: self.model.generalized_latency(mu), "generalizedLatency")
        self.function_test(lambda: self.model.as_dsl("TestName"), "convert_to_DSL", sort = True)


    def incorrect_behavior_tests(self):
        """Testing behaviors that are expected to fail."""
        mu = self.mu
        if mu is not None:
            x0 = parse_initial_state(self.args, self.model.number_of_initial_tokens())
            self.incorrect_test(lambda: self.model.latency(x0, mu-Fraction(0.01)), \
                                "The requested period mu is smaller than smallest period the " \
                                    "system can sustain. Therefore, it has no latency.")
            self.incorrect_test(lambda: self.model.generalized_latency(mu-Fraction(0.01)), \
                                "The requested period mu is smaller than smallest period the " \
                                    "system can sustain. Therefore, it has no latency.")

    class Namespace:
        """Class to create namespace args consisting of user input arguments"""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


@pytest.mark.parametrize("test_model", MODEL_SDF_FILES)
def test_sdf(test_model: str):
    """Test an SDF model."""
    m = SDFPyTest(test_model)
    m.correct_behavior_tests()
    m.incorrect_behavior_tests()
    m.write_output_file()


###############################################
### Max-Plus Models                         ###
###############################################

class MPMPyTest(ModelPytest):
    """Test an MPM model."""
    def __init__(self, model):

        # First load LTL model
        self.model_name = model[:-4]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".mpm")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")

        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r', encoding='utf-8') as mpm_file:
            dsl = mpm_file.read()
        self.name, self.matrices, self.vector_sequences, \
            self.event_sequences = MaxPlusMatrixModel.from_dsl(dsl)

        sq = ",".join(self.event_sequences.keys())
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

    def correct_behavior_tests(self):
        """Test behaviors that are expected to succeed."""
        # Function used to obtain multiple results
        mat = get_square_matrix(self.matrices, self.args)

        # Deterministic results
        self.function_test(self.matrices[mat].eigenvalue, "Eigenvalues")
        self.function_test(self.matrices[mat].star_closure, "starClosure")

        # Only check convolution when sequence is available
        if len(self.args.sequences) > 1:  # type: ignore
            sequences, res = _convolution(self.event_sequences, self.args)
            self.function_test(lambda: sequences, "convolution_sequences")
            self.function_test(lambda: vars(res)['_sequence'], "convolution_res")
            res = self.event_sequences[list(self.event_sequences.keys())[0]].delay(self.args.numberofiterations)  # type: ignore
            self.function_test(lambda: vars(res)['_sequence'], "delay_sequence")
            res = self.event_sequences[list(self.event_sequences.keys())[0]].scale(self.args.numberofiterations)  # type: ignore
            self.function_test(lambda: vars(res)['_sequence'], "scale_sequence")
            sequences, res = _maximum(self.event_sequences, self.args)
            self.function_test(lambda: sequences, "maximum_analysis_sequences")
            self.function_test(lambda: vars(res)['_sequence'], "maximum_analysis_res")
        _, cl = self.matrices[list(self.matrices.keys())[0]].star_closure()
        # if success:
        self.function_test(lambda: cl, "star_closure")
        matrix = self.matrices[list(self.matrices.keys())[0]]
        self.function_test(lambda: vars(MaxPlusMatrixModel.multiply_sequence([matrix]))["_rows"], \
                           "multiply")

        # Non deterministic results, False added as argument
        # Eigenvectors can appear in different order
        self.function_test(self.matrices[mat].eigenvectors, "Eigenvectors", False)

    def incorrect_behavior_tests(self):
        """Test behaviors that are expected to fail."""

    class Namespace:
        """Class to create namespace args consisting of user input arguments"""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

@pytest.mark.parametrize("test_model", MODEL_MPM_FILES)
def test_mpm(test_model: str):
    """Test an MPM model."""
    m = MPMPyTest(test_model)
    m.correct_behavior_tests()
    m.incorrect_behavior_tests()
    m.write_output_file()
