"""Testing Markov Chain package"""

import os
from typing import Dict, Optional, Union
import pytest
from modeltest.modeltest import Model_pytest # Import Model_pytest class from modeltest package
from markovchains.libdtmc import MarkovChain
from markovchains.utils.utils import MarkovChainException, sort_names
from markovchains.utils.statistics import Statistics, DistributionStatistics, StopConditions

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
        with open(self.model_loc, 'r', encoding='utf-8') as dtmcFile:
            dsl = dtmcFile.read()
        dName, dModel = MarkovChain.from_dsl(dsl)

        if dModel is None or dName is None:
            raise MarkovChainException("Failed to read test model.")

        self.name, self.model = dName, dModel


        # Store default recurrent state
        _,recurrentStates = self.model.classify_transient_recurrent()
        self.state = sort_names(recurrentStates)[0]

        # Set seed for markovchain simulation functions
        #   When adding a function in behavior_tests which relies on seed will corrupt following functions
        #   Adding function relying on seed requires deleting .json files in output folder
        self.model.set_seed(0)

        # Set recurrent state to be the first in the trace
        self.model.set_recurrent_state(None)

    def statisticsAndStop(self, s: Statistics, stop: Optional[str]):
        return s.confidence_interval(), s.ab_error(), s.re_error(), s.mean_estimate(), s.std_dev_estimate(), stop

    def dictionaryStatisticsAndStop(self, s: Optional[Dict[str,Statistics]], stop: Union[str,Dict[str,str]]):
        if s is None:
            return None
        res = dict()
        for t in s.keys():
            res[t] = [s[t].confidence_interval(), s[t].ab_error(), s[t].re_error(), s[t].mean_estimate(), s[t].std_dev_estimate()]
        return res, stop

    def distributionStatisticsAndStop(self, s: Optional[DistributionStatistics], stop: Optional[str]):
        if s is None:
            return "None"
        return s.confidence_intervals(), s.ab_error(), s.re_error(), s.point_estimates(), s.std_dev_estimates(), stop

    def Correct_behavior_tests(self):
        self.function_test(lambda: self.model.states(), "states")
        self.function_test(lambda: self.model.reward_vector(), "rewardVector")
        self.function_test(lambda: self.model.execute_steps(0), "executeSteps_0")
        self.function_test(lambda: self.model.execute_steps(15), "executeSteps_15")
        self.function_test(lambda: self.model.classify_transient_recurrent(), "classifyTransientRecurrent", sort = True)
        self.function_test(lambda: self.model.classify_periodicity(), "classifyPeriodicity")
        self.function_test(lambda: self.model.determine_mc_type(), "determineMCType")
        self.function_test(lambda: self.model.hitting_probabilities(self.state), "hittingProbabilities")
        self.function_test(lambda: self.model.limiting_matrix(), "limitingMatrix")
        self.function_test(lambda: self.model.limiting_distribution(), "limitingDistribution")
        self.function_test(lambda: self.model.long_run_reward(), "longRunReward")
        N=0
        trace = self.model.execute_steps(N)
        for k in range(N+1):
            self.function_test(lambda: self.model.reward_for_distribution(trace[k]), "transient_reward_0_step_"+str(k))
        N=10
        trace = self.model.execute_steps(N)
        for k in range(N+1):
            self.function_test(lambda: self.model.reward_for_distribution(trace[k]), "transient_reward_10_step_"+str(k))
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.model.markov_trace(0), "markovTrace_0")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.model.markov_trace(15), "markovTrace_15")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.statisticsAndStop(*self.model.long_run_expected_average_reward(StopConditions(0.95,-1,-1,-1,1,-1))), "longRunExpectedAverageReward_cycle")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.statisticsAndStop(*self.model.long_run_expected_average_reward(StopConditions(0.95,-1,-1,1,-1,-1))), "longRunExpectedAverageReward_steps")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.statisticsAndStop(*self.model.long_run_expected_average_reward(StopConditions(0.95,0.5,-1,-1,-1,-1))), "longRunExpectedAverageReward_abs")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.statisticsAndStop(*self.model.long_run_expected_average_reward(StopConditions(0.95,-1,0.5,1000,-1,-1))), "longRunExpectedAverageReward_rel")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.distributionStatisticsAndStop(*self.model.cezaro_limit_distribution(StopConditions(0.95,-1,-1,-1,1,-1))), "cezaroLimitDistribution_cycle")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.distributionStatisticsAndStop(*self.model.cezaro_limit_distribution(StopConditions(0.95,-1,-1,1,-1,-1))), "cezaroLimitDistribution_steps")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.distributionStatisticsAndStop(*self.model.cezaro_limit_distribution(StopConditions(0.95,0.5,-1,-1,-1,-1))), "cezaroLimitDistribution_abs")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.distributionStatisticsAndStop(*self.model.cezaro_limit_distribution(StopConditions(0.95,-1,0.5,1000,-1,-1))), "cezaroLimitDistribution_rel")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.statisticsAndStop(*self.model.estimation_expected_reward(StopConditions(0.95,-1,-1,-1,1,-1), 1)), "estimationExpectedReward_step")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.statisticsAndStop(*self.model.estimation_expected_reward(StopConditions(0.95,0.5,-1,-1,-1,-1), 1)), "estimationExpectedReward_abs")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.statisticsAndStop(*self.model.estimation_expected_reward(StopConditions(0.95,-1,0.5,-1,1000,-1), 1)), "estimationExpectedReward_rel")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.distributionStatisticsAndStop(*self.model.estimation_transient_distribution(StopConditions(0.95,-1,-1,1,1,-1), 1)), "estimationDistribution")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.dictionaryStatisticsAndStop(*self.model.estimation_hitting_probability_state(StopConditions(0.95,-1,-1,1,1,-1), self.state, self.model.states())), "estimationHittingState")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.dictionaryStatisticsAndStop(*self.model.estimation_reward_until_hitting_state(StopConditions(0.95,-1,-1,1,1,30), self.state, self.model.states())), "estimationHittingreward")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.dictionaryStatisticsAndStop(*self.model.estimation_hitting_probability_state_set(StopConditions(0.95,-1,-1,1,1,-1), [self.state], self.model.states())), "estimationHittingStateSet")
        self.model.set_recurrent_state(None) # reset trace recurrent state
        self.function_test(lambda: self.dictionaryStatisticsAndStop(*self.model.estimation_reward_until_hitting_state_set(StopConditions(0.95,-1,-1,1,1,30), [self.state], self.model.states())), "estimationHittingRewardSet")
        self.function_test(lambda: self.model.as_dsl("TestName"), "convert_to_DSL", sort = True)


    def Incorrect_behavior_tests(self):
        self.incorrect_test(lambda: self.model.execute_steps(-2), 'Number of steps must be non-negative.')
        self.incorrect_test(lambda: self.model.hitting_probabilities('NOT_A_STATE'), "'NOT_A_STATE'")
        self.incorrect_test(lambda: self.model.long_run_expected_average_reward(StopConditions(-1.00,0,0,1000,0,-1)), "p must be in the range 0.0 < p < 1.0")
        self.incorrect_test(lambda: self.model.cezaro_limit_distribution(StopConditions(-1.00,0,0,1000,0,-1.-1)), "p must be in the range 0.0 < p < 1.0")
        self.incorrect_test(lambda: self.model.estimation_expected_reward(StopConditions(-1.00,0,0,1000,0,-1), 1), "p must be in the range 0.0 < p < 1.0")
        self.incorrect_test(lambda: self.model.estimation_reward_until_hitting_state_set(StopConditions(-1.00,0,0,1000,0,-1), [self.state], self.model.states()), "p must be in the range 0.0 < p < 1.0")


@pytest.mark.parametrize("test_model", MODEL_FILES)
def test_model(test_model: str):
    m = Markov_pytest(test_model)
    m.Correct_behavior_tests()
    m.Incorrect_behavior_tests()
    m.write_output_file()


