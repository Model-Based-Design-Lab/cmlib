import os
import pytest
from modeltest.modeltest import Model_pytest # Import Model_pytest class from modeltest package
from markovchains.libdtmc import MarkovChain
from markovchains.utils.utils import sortNames

# Collect models and output files
TEST_FILE_FOLDER = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(TEST_FILE_FOLDER, "models")
OUTPUT_FOLDER = os.path.join(TEST_FILE_FOLDER, "output")
MODEL_FILES = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".dtmc")]

class Markov_pytest(Model_pytest):
    
    model: MarkovChain
    state: str

    def __init__(self, model):

        # First load markov chain model
        self.model_name = model[:-5]
        self.model_loc = os.path.join(MODEL_FOLDER, self.model_name + ".dtmc")
        self.output_loc = os.path.join(OUTPUT_FOLDER, self.model_name + ".json")

        super().__init__(self.output_loc)

        # Open model
        with open(self.model_loc, 'r') as dtmcFile:
            dsl = dtmcFile.read()
        dName, dModel = MarkovChain.fromDSL(dsl)

        if dModel is None or dName is None:
            raise Exception("Failed to read test model.")
        
        self.name, self.model = dName, dModel


        # Store default recurrent state
        _,recurrentStates = self.model.classifyTransientRecurrent()
        self.state = sortNames(recurrentStates)[0]

        # Set seed for markovchain simulation functions
        #   When adding a function in behaviour_tests which relies on seed will corrupt following functions
        #   Adding function relying on seed requires deleting .json files in output folder
        self.model.setSeed(0)

        # Set recurrent state to be the first in the trace
        self.model.setRecurrentState(None)
        
    def Correct_behaviour_tests(self):
        self.function_test(lambda: self.model.states(), "states")
        self.function_test(lambda: self.model.rewardVector(), "rewardVector")
        self.function_test(lambda: self.model.executeSteps(0), "executeSteps_0")
        self.function_test(lambda: self.model.executeSteps(15), "executeSteps_15")
        self.function_test(lambda: self.model.classifyTransientRecurrent(), "classifyTransientRecurrent", sort = True)
        self.function_test(lambda: self.model.classifyPeriodicity(), "classifyPeriodicity")
        self.function_test(lambda: self.model.determineMCType(), "determineMCType")
        self.function_test(lambda: self.model.hittingProbabilities(self.state), "hittingProbabilities")
        self.function_test(lambda: self.model.limitingMatrix(), "limitingMatrix")
        self.function_test(lambda: self.model.limitingDistribution(), "limitingDistribution")
        self.function_test(lambda: self.model.longRunReward(), "longRunReward")
        N=0
        trace = self.model.executeSteps(N)
        for k in range(N+1):
            self.function_test(lambda: self.model.rewardForDistribution(trace[k]), "transient_reward_0_step_"+str(k))
        N=10
        trace = self.model.executeSteps(N)
        for k in range(N+1):
            self.function_test(lambda: self.model.rewardForDistribution(trace[k]), "transient_reward_10_step_"+str(k))
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.markovTrace(0), "markovTrace_0")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.markovTrace(15), "markovTrace_15")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.longRunExpectedAverageReward((0.95,-1,-1,-1,1,-1)), "longRunExpectedAverageReward_cycle")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.longRunExpectedAverageReward((0.95,-1,-1,1,-1,-1)), "longRunExpectedAverageReward_steps")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.longRunExpectedAverageReward((0.95,0.5,-1,-1,-1,-1)), "longRunExpectedAverageReward_abs")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.longRunExpectedAverageReward((0.95,-1,0.5,1000,-1,-1)), "longRunExpectedAverageReward_rel")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.cezaroLimitDistribution((0.95,-1,-1,-1,1,-1)), "cezaroLimitDistribution_cycle")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.cezaroLimitDistribution((0.95,-1,-1,1,-1,-1)), "cezaroLimitDistribution_steps")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.cezaroLimitDistribution((0.95,0.5,-1,-1,-1,-1)), "cezaroLimitDistribution_abs")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.cezaroLimitDistribution((0.95,-1,0.5,1000,-1,-1)), "cezaroLimitDistribution_rel")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationExpectedReward((0.95,-1,-1,-1,1,-1), 1), "estimationExpectedReward_step")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationExpectedReward((0.95,0.5,-1,-1,-1,-1), 1), "estimationExpectedReward_abs")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationExpectedReward((0.95,-1,0.5,-1,1000,-1), 1), "estimationExpectedReward_rel")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationTransientDistribution((0.95,-1,-1,1,1,-1), 1), "estimationDistribution")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationHittingProbabilityState((0.95,-1,-1,1,1,-1), self.state, self.model.states()), "estimationHittingState")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationRewardUntilHittingState((0.95,-1,-1,1,1,-1), self.state, self.model.states()), "estimationHittingreward")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationHittingProbabilityStateSet((0.95,-1,-1,1,1,-1), [self.state], self.model.states()), "estimationHittingStateSet")
        self.model.setRecurrentState(None) # reset trace recurrent state
        self.function_test(lambda: self.model.estimationRewardUntilHittingStateSet((0.95,-1,-1,1,1,-1), [self.state], self.model.states()), "estimationHittingRewardSet")

    def Incorrect_behaviour_tests(self):
        self.incorrect_test(lambda: self.model.executeSteps(-2), 'Number of steps must be non-negative.')
        self.incorrect_test(lambda: self.model.hittingProbabilities('NOT_A_STATE'), "'NOT_A_STATE'")
        self.incorrect_test(lambda: self.model.longRunExpectedAverageReward((-1.00,0,0,1000,0,-1)), "p must be in the range 0.0 < p < 1.0")
        self.incorrect_test(lambda: self.model.cezaroLimitDistribution((-1.00,0,0,1000,0,-1.-1)), "p must be in the range 0.0 < p < 1.0")
        self.incorrect_test(lambda: self.model.estimationExpectedReward((-1.00,0,0,1000,0,-1), 1), "p must be in the range 0.0 < p < 1.0")
        self.incorrect_test(lambda: self.model.estimationRewardUntilHittingStateSet((-1.00,0,0,1000,0,-1), [self.state], self.model.states()), "p must be in the range 0.0 < p < 1.0")


@pytest.mark.parametrize("test_model", MODEL_FILES)
def test_model(test_model: str):
    m = Markov_pytest(test_model)
    m.Correct_behaviour_tests()
    m.Incorrect_behaviour_tests()
    m.write_output_file()

 
