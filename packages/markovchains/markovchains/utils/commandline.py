
'''Operations on Markov chains '''

import argparse
from typing import Any, Dict, List, Optional, Union

from markovchains.libdtmc import MarkovChain
from markovchains.utils.graphs import plotSvg
from markovchains.utils.linalgebra import matPower, TVector
from markovchains.utils.statistics import StopConditions
from markovchains.utils.utils import sort_names, string_to_float, stop_criteria, nr_of_steps, print_sorted_list, print_sorted_set, print_list_of_strings, print_optional_interval, print_optional_list, print_optional_list_of_intervals, pretty_print_matrix, pretty_print_vector, pretty_print_value, optional_float_or_string_to_string, print_table
import markovchains.utils.linalgebra as linalg


from markovchains.utils.operations import MarkovChainOperations, OperationDescriptions, OP_DTMC_CLASSIFY_TRANSIENT_RECURRENT, OP_DTMC_COMMUNICATINGSTATES, OP_DTMC_EXECUTION_GRAPH, OP_DTMC_LIST_RECURRENT_STATES, OP_DTMC_LIST_STATES, OP_DTMC_LIST_TRANSIENT_STATES, OP_DTMC_MC_TYPE, OP_DTMC_PERIODICITY, OP_DTMC_TRANSIENT, OP_DTMC_CEZARO_LIMIT_DISTRIBUTION, OP_DTMC_ESTIMATION_DISTRIBUTION, OP_DTMC_ESTIMATION_EXPECTED_REWARD, OP_DTMC_ESTIMATION_HITTING_REWARD, OP_DTMC_ESTIMATION_HITTING_REWARD_SET, OP_DTMC_ESTIMATION_HITTING_STATE, OP_DTMC_ESTIMATION_HITTING_STATE_SET, OP_DTMC_HITTING_PROBABILITY, OP_DTMC_HITTING_PROBABILITY_SET, OP_DTMC_LIMITING_DISTRIBUTION, OP_DTMC_LIMITING_MATRIX, OP_DTMC_LONG_RUN_EXPECTED_AVERAGE_REWARD, OP_DTMC_LONG_RUN_REWARD, OP_DTMC_MARKOV_TRACE, OP_DTMC_REWARD_TILL_HIT, OP_DTMC_REWARD_TILL_HIT_SET, OP_DTMC_TRANSIENT_MATRIX, OP_DTMC_TRANSIENT_REWARDS
import sys



def main():

    # optional help flag explaining usage of each individual operation
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('-oh', '--operationhelp', dest='opHelp', nargs="?", const=" ")
    options, remainder = parser.parse_known_args() # Only use options of parser above

    if options.opHelp: # Check if -oh has been called
        if options.opHelp not in MarkovChainOperations:
            print("Operation '{}' does not exist. List of operations:\n\t- {}".format(options.opHelp, "\n\t- ".join(MarkovChainOperations)))
        else:
            print("{}: {}".format(options.opHelp, OperationDescriptions[MarkovChainOperations.index(options.opHelp)]))
        exit(1)

    parser = argparse.ArgumentParser(
        description='Perform operations on discrete-time Markov chains.\nhttps://computationalmodeling.info')
    parser.add_argument('markovchain', help="the Markov chain to analyze")
    parser.add_argument('-op', '--operation', dest='operation',
                        help="the operation or analysis to perform, one of : {}.\nUse 'markovchains -oh OPERATION' for information about the specific operation.".format("; \n".join(MarkovChainOperations)))
    parser.add_argument('-ns', '--numberofsteps', dest='numberOfSteps',
                        help="the number of steps to execute")
    parser.add_argument('-s', '--state', dest='targetState',
                        help="the state for the operation")
    parser.add_argument('-ss', '--stateset', dest='targetStateSet',
                        help="the set of state for the operation as a non-empty comma-separated list")
    parser.add_argument('-r', '--rewardset', dest='stateRewardSet',
                        help="the set of state reward for the operation as a non-empty comma-separated list")
    parser.add_argument('-sa', '--startingset', dest='stateStartingSet',
                        help="the set of starting states for the simulation hitting operations as a non-empty comma-separated list")
    parser.add_argument('-c', '--conditions', dest='Conditions',
                        help="The stop conditions for simulating the markovchain [confidence,abError,reError,numberOfSteps,numberOfPaths,timeInSeconds]")
    parser.add_argument('-sd', '--seed', dest='Seed',
                        help="Simulation seed for pseudo random variables")

    args = parser.parse_args(remainder)

    if args.operation not in MarkovChainOperations:
        sys.stderr.write("Unknown operation: {}\n".format(args.operation))
        exit(1)

    dsl:str = ""

    if args.markovchain:
        try:
            with open(args.markovchain, 'r', encoding='utf-8') as dtmcFile:
                dsl = dtmcFile.read()
        except FileNotFoundError as e:
            sys.stderr.write("File does not exist: {}\n.".format(args.markovchain))
            exit(1)

    try:
        process(args, dsl)
    except Exception as e:
        sys.stderr.write("{}\n".format(e))
        # raise e
        exit(1)

    exit(0)


def requireNumberOfSteps(args: Any)->int:
    if args.numberOfSteps is None:
        raise Exception("number of steps (-ns option) must be specified.")
    try:
        ns: int = nr_of_steps(int(args.numberOfSteps))
    except Exception as e:
        raise Exception("Failed to determine number of steps.")
    if ns < 0:
        raise Exception("Number of steps must be a non-negative number.")
    return ns

def requireTargetState(M: MarkovChain, args: Any)->str:
    if args.targetState is None:
        raise Exception("A target state must be specified with the -s option.")
    s: str = args.targetState
    if not s in M.states():
            raise(Exception("The specified target state {} does not exist.".format(s)))
    return args.targetState

def requireTargetStateSet(M: MarkovChain, args: Any)->List[str]:
    if args.targetStateSet is None:
        raise Exception("A target state set must be specified with the -ss option.")
    stateSet = [s.strip() for s in args.targetStateSet.split(',')]
    for s in stateSet:
        if not s in M.states():
            raise(Exception("State {} in specified state set does not exist.".format(s)))
    return stateSet

def requireStopCriteria(args: Any)->StopConditions:
    if args.Conditions is None:
        raise Exception("Stop conditions must be specified with the -c option.")
    cc = stop_criteria([string_to_float(i, -1.0) for i in args.Conditions[1:-1].split(',')])
    return cc

def setSeed(args: Any, M: MarkovChain):
    if args.Seed is not None:
        M.setSeed(int(args.Seed))

def setStartingStateSet(args: Any, M: MarkovChain):
    S: List[str]
    if args.stateStartingSet is not None:
        S = [s.strip() for s in args.stateStartingSet.split(',')]
    else:
        S = M.states()
    return S


def requireMarkovChain(M: Optional[MarkovChain]) -> MarkovChain:
    if M is None:
        raise Exception("A Markov Chain is needed.")
    return M

def process(args, dsl):

    operation = args.operation

    M = None

    if operation in MarkovChainOperations:
        _, M = MarkovChain.fromDSL(dsl)
        if M is None:
            exit(1)

    # let the type checker know that we certainly have a Markov Chain from here
    M = requireMarkovChain(M)

    # just list all states
    if operation == OP_DTMC_LIST_STATES:
        res = M.states()
        print_sorted_list(res)

    # list the recurrent states
    if operation == OP_DTMC_LIST_RECURRENT_STATES:
        _, recurrentStates = M.classifyTransientRecurrent()
        print_sorted_list(recurrentStates)

    # list the transient states
    if operation == OP_DTMC_LIST_TRANSIENT_STATES:
        trans, _ = M.classifyTransientRecurrent()
        print_sorted_list(trans)

    # create graph for a number of steps
    if operation == OP_DTMC_EXECUTION_GRAPH:
        N = requireNumberOfSteps(args)
        trace = linalg.transpose(M.executeSteps(N))
        states = M.states()
        data = dict()
        data['k'] = range(0,N+1)
        k = 0
        for s in sort_names(states):
            data[s] = trace[k]
            k += 1
        print(plotSvg(data, states))

    # determine classes of communicating states
    if operation == OP_DTMC_COMMUNICATINGSTATES:
        print("Classes of communicating states:")
        for s in M.communicatingClasses():
            print_sorted_set(s)

    # classify transient and recurrent states
    if operation == OP_DTMC_CLASSIFY_TRANSIENT_RECURRENT:
        trans, recurrentStates = M.classifyTransientRecurrent()
        print("Transient states:")
        print_sorted_set(trans)
        print("Recurrent states:")
        print_sorted_set(recurrentStates)

    # classify transient and recurrent states
    if operation == OP_DTMC_PERIODICITY:
        per = M.classifyPeriodicity()
        print("The set of aperiodic recurrent states is:")
        aperStates =  [s for s in per.keys() if per[s] == 1]
        print_sorted_set(aperStates)

        if len(aperStates) < len(per):
            periodicities = set(per.values())
            if 1 in periodicities:
                periodicities.remove(1)
            for p in periodicities:
                print("The set of periodic recurrent states with periodicity {} is.".format(p))
                pPeriodicStates =  [s for s in per.keys() if per[s] == p]
                print_sorted_set(pPeriodicStates)

    # classify transient and recurrent states
    if operation == OP_DTMC_MC_TYPE:
        mcType = M.determineMCType()
        print("The type of the MC is: {}".format(mcType))

    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT:
        N = requireNumberOfSteps(args)
        trace = M.executeSteps(N)
        states = M.states()

        print("Transient analysis:\n")
        print ("State vector:")
        print_list_of_strings(states)

        for k in range(N+1):
            print("Step {}:".format(k))
            print("Distribution: ", end="")
            pretty_print_vector(trace[k])

    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT_REWARDS:
        N = requireNumberOfSteps(args)
        trace = M.executeSteps(N)

        print("Transient reward analysis:")
        for k in range(N+1):
            print("\nStep {}:".format(k))
            print("Expected Reward: ", end='')
            pretty_print_value(M.rewardForDistribution(trace[k]))


    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT_MATRIX:
        N = requireNumberOfSteps(args)
        mat = M.transitionMatrix()

        print ("State vector:")
        print_list_of_strings(M.states())
        print("Transient analysis:\n")
        print("Matrix for {} steps:\n".format(N))
        pretty_print_matrix(matPower(mat, N))

    if operation == OP_DTMC_LIMITING_MATRIX:
        mat = M.limitingMatrix()
        print ("State vector:")
        print_list_of_strings(M.states())
        print ("Limiting Matrix:\n")
        pretty_print_matrix(mat)

    if operation == OP_DTMC_LIMITING_DISTRIBUTION:
        lDist: TVector = M.limitingDistribution()

        print ("State vector:")
        print_list_of_strings(M.states())
        print ("Limiting Distribution:")
        pretty_print_vector(lDist)

    if operation == OP_DTMC_LONG_RUN_REWARD:
        mcType = M.determineMCType()
        r = M.longRunReward()
        if 'non-ergodic' in mcType:
            print("The long-run expected average reward is: {}\n".format(r))
        else:
            print("The long-run expected reward is: {}\n".format(r))

    if operation == OP_DTMC_HITTING_PROBABILITY:
        s = requireTargetState(M, args)
        prob = M.hittingProbabilities(s)
        print("The hitting probabilities for {} are:".format(s))
        for t in sort_names(M.states()):
            print("f({}, {}) = {}".format(t, s, prob[t]))

    if operation == OP_DTMC_REWARD_TILL_HIT:
        s = requireTargetState(M, args)
        res = M.rewardTillHit(s)
        print("The expected rewards until hitting {} are:".format(s))
        for s in sort_names(res.keys()):
            print("From state {}: {}".format(s, res[s]))

    if operation == OP_DTMC_HITTING_PROBABILITY_SET:
        targetStateSet = requireTargetStateSet(M, args)
        prob = M.hittingProbabilitiesSet(targetStateSet)
        print("The hitting probabilities for {{{}}} are:".format(', '.join(prob)))
        ss = ', '.join(targetStateSet)
        for t in sort_names(M.states()):
            print("f({}, {{{}}}) = {}".format(t, ss, prob[t]))

    if operation == OP_DTMC_REWARD_TILL_HIT_SET:
        s = requireTargetStateSet(M, args)
        res = M.rewardTillHitSet(s)
        print("The expected rewards until hitting {{{}}} are:".format(', '.join(s)))
        for t in sort_names(res.keys()):
            print("From state {}: {}".format(t, res[t]))

    if operation == OP_DTMC_MARKOV_TRACE:
        setSeed(args, M)
        N = requireNumberOfSteps(args)
        trace = M.markovTrace(N)
        print("{}".format(trace))

    if operation == OP_DTMC_LONG_RUN_EXPECTED_AVERAGE_REWARD:
        setSeed(args, M)
        if args.targetState:
            M.setRecurrentState(args.targetState)
        C = requireStopCriteria(args)
        statistics, stop = M.longRunExpectedAverageReward(C)
        if statistics.cycle_count() == 0:
            print("Recurrent state has not been reached, no realizations found")
        else:
            print("Simulation termination reason: {}".format(stop))
            print("The long run expected average reward is:")
            print("\tEstimated mean: {}".format(optional_float_or_string_to_string(statistics.mean_estimate_result())))
            print("\tConfidence interval: {}".format(print_optional_interval(statistics.confidence_interval())))
            print("\tAbsolute error bound: {}".format(optional_float_or_string_to_string(statistics.ab_error())))
            print("\tRelative error bound: {}".format(optional_float_or_string_to_string(statistics.re_error())))
            print("\tNumber of cycles: {}".format(statistics.cycle_count()))

    if operation == OP_DTMC_CEZARO_LIMIT_DISTRIBUTION:
        setSeed(args, M)
        if args.targetState:
            M.setRecurrentState(args.targetState)
        C = requireStopCriteria(args)
        distributionStatistics, stop = M.cezaroLimitDistribution(C)

        if distributionStatistics is None:
            print("Recurrent state has not been reached, no realizations found")
        else:
            print("Simulation termination reason: {}".format(stop))
            print("Cezaro limit distribution: {}".format(print_optional_list(distributionStatistics.point_estimates(), "Could not be determined")))
            print("Number of cycles: {}\n".format(distributionStatistics.cycle_count()))
            dist: Optional[List[float]] = distributionStatistics.point_estimates()
            if dist is not None:
                intervals = distributionStatistics.confidence_intervals()
                abError = distributionStatistics.ab_error()
                reError = distributionStatistics.re_error()
                states: List[str] = M.states()
                for i in range(len(states)):
                    print("[{}]: {:.4f}".format(states[i], dist[i]))
                    print("\tConfidence interval: {}".format(print_optional_interval(None if intervals is None else intervals[i])))
                    print("\tAbsolute error bound: {}".format(optional_float_or_string_to_string(abError[i])))
                    print("\tRelative error bound: {}".format(optional_float_or_string_to_string(reError[i])))
                    print("\n")

    if operation == OP_DTMC_ESTIMATION_EXPECTED_REWARD:
        setSeed(args, M)
        N = requireNumberOfSteps(args)
        C = requireStopCriteria(args)
        statistics, stop = M.estimationExpectedReward(C, N)
        print("Simulation termination reason: {}".format(stop))
        print("\tExpected reward: {}".format(optional_float_or_string_to_string(statistics.mean_estimate_result())))
        print("\tConfidence interval: {}".format(print_optional_interval(statistics.confidence_interval())))
        print("\tAbsolute error bound: {}".format(optional_float_or_string_to_string(statistics.ab_error())))
        print("\tRelative error bound: {}".format(optional_float_or_string_to_string(statistics.re_error())))
        print("\tNumber of realizations: ", statistics.cycle_count())

    if operation == OP_DTMC_ESTIMATION_DISTRIBUTION:
        setSeed(args, M)
        states = M.states()
        N = requireNumberOfSteps(args)
        C = requireStopCriteria(args)
        distributionStatistics, stop = M.estimationTransientDistribution(C, N)
        print("Simulation termination reason: {}".format(stop))
        print("The estimated distribution after {} steps of [{}] is as follows:".format(N, ", ".join(states)))
        print("\tDistribution: {}".format(print_optional_list(distributionStatistics.point_estimates())))
        print("\tConfidence intervals: {}".format(print_optional_list_of_intervals(distributionStatistics.confidence_intervals())))
        print("\tAbsolute error bound: {}".format(optional_float_or_string_to_string(distributionStatistics.max_ab_error())))
        print("\tRelative error bound: {}".format(optional_float_or_string_to_string(distributionStatistics.max_re_error())))
        print("\tNumber of realizations: ", distributionStatistics.cycle_count())

    if operation == OP_DTMC_ESTIMATION_HITTING_STATE:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetState(M, args)
        C = requireStopCriteria(args)
        statisticsDict, stop = M.estimationHittingProbabilityState(C, s, S)
        if statisticsDict is None:
            print("A timeout has occurred during the analysis.")
        else:
            dStop: Dict[str,str] = stop  # type: ignore
            print("Estimated hitting probabilities for {} are:".format(s))
            tableHS: List[Union[str,List[str]]] = []
            for i, t in enumerate(S):
                statistics = statisticsDict[t]
                tableHS.append([
                    "f({}, {}) = {}".format(t, s, optional_float_or_string_to_string(statistics.mean_estimate_result())),
                    "int: {}".format(print_optional_interval(statistics.confidence_interval())),
                    "abEr: {}".format(optional_float_or_string_to_string(statistics.ab_error())),
                    "reEr: {}".format(optional_float_or_string_to_string(statistics.re_error())),
                    "#paths: {}".format(statistics.nr_paths()),
                    "stop: {}".format(dStop[t])
                ])
            print_table(tableHS, 4)

    if operation == OP_DTMC_ESTIMATION_HITTING_REWARD:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetState(M, args)
        C = requireStopCriteria(args)
        statisticsDict, stop = M.estimationRewardUntilHittingState(C, s, S)
        if statisticsDict is None:
            print("A timeout has occurred during the analysis.")
        else:
            print("Estimated cumulative reward until hitting {} are:".format(s))
            tableRS: List[Union[str,List[str]]] = []
            for i, t in enumerate(S):
                statistics = statisticsDict[t]
                if not isinstance(statistics.mean_estimate_result(), float):
                    tableRS.append("From state {}: {}".format(S[i], optional_float_or_string_to_string(statistics.mean_estimate_result())))
                else:
                    dStop: Dict[str,str] = stop  # type: ignore
                    tableRS.append([
                        "From state {}: {}".format(S[i], optional_float_or_string_to_string(statistics.mean_estimate_result())),
                        "int: {}".format(print_optional_interval(statistics.confidence_interval())),
                        "abEr: {}".format(optional_float_or_string_to_string(statistics.ab_error())),
                        "reEr: {}".format(optional_float_or_string_to_string(statistics.re_error())),
                        "#paths: {}".format(statistics.nr_paths()),
                        "stop: {}".format(dStop[t])
                    ])
            print_table(tableRS, 4)

    if operation == OP_DTMC_ESTIMATION_HITTING_STATE_SET:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetStateSet(M, args)
        C = requireStopCriteria(args)
        statisticsDict, stop = M.estimationHittingProbabilityStateSet(C, s, S)
        if statisticsDict is None:
            print("A timeout has occurred during the analysis.")
        else:
            print("Estimated hitting probabilities for {{{}}} are:".format(', '.join(s)))
            table: List[Union[str,List[str]]] = []
            for i, t in enumerate(S):
                statistics = statisticsDict[t]
                if not isinstance(statistics.mean_estimate_result(), float):
                    table.append("From state {}: {}".format(S[i], statistics.mean_estimate_result()))
                else:
                    dStop: Dict[str,str] = stop  # type: ignore
                    table.append([
                        "f({}, {{{}}}) = {}".format(S[i], ', '.join(s), optional_float_or_string_to_string(statistics.mean_estimate_result())),
                        "int: {}".format(print_optional_interval(statistics.confidence_interval())),
                        "abEr: {}".format(optional_float_or_string_to_string(statistics.ab_error())),
                        "reEr: {}".format(optional_float_or_string_to_string(statistics.re_error())),
                        "#paths: {}".format(statistics.nr_paths()),
                        "stop: {}".format(dStop[t])
                    ])
            print_table(table, 4)


    if operation == OP_DTMC_ESTIMATION_HITTING_REWARD_SET:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetStateSet(M, args)
        C = requireStopCriteria(args)
        statisticsDict, stop = M.estimationRewardUntilHittingStateSet(C, s, S)
        if statisticsDict is None:
            print("A timeout has occurred during the analysis.")
        else:
            print("Estimated cumulative reward until hitting {{{}}} are:".format(', '.join(s)))
            tableRSet: List[Union[str,List[str]]] = []
            for i, t in enumerate(S):
                statistics = statisticsDict[t]
                if not isinstance(statistics.mean_estimate_result(), float):
                    tableRSet.append("From state {}: {}".format(S[i], optional_float_or_string_to_string(statistics.mean_estimate_result())))
                else:
                    dStop: Dict[str,str] = stop  # type: ignore
                    tableRSet.append([
                        "From state {}: {}".format(S[i], optional_float_or_string_to_string(statistics.mean_estimate_result())),
                        "int: {}".format(print_optional_interval(statistics.confidence_interval())),
                        "abEr: {}".format(optional_float_or_string_to_string(statistics.ab_error())),
                        "reEr: {}".format(optional_float_or_string_to_string(statistics.re_error())),
                        "#paths: {}".format(statistics.nr_paths()),
                        "stop: {}".format(dStop[t])
                    ])
            print_table(tableRSet, 4)

if __name__ == "__main__":
    main()