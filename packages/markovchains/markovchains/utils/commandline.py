
'''Operations on Markov chains '''

import argparse
import sys
from typing import Any, Dict, List, Optional, Union

import markovchains.utils.linalgebra as linalg
from markovchains.libdtmc import MarkovChain
from markovchains.utils.graphs import plotSvg
from markovchains.utils.linalgebra import TVector, matPower
from markovchains.utils.operations import (
    OP_DTMC_CEZARO_LIMIT_DISTRIBUTION, OP_DTMC_CLASSIFY_TRANSIENT_RECURRENT,
    OP_DTMC_COMMUNICATINGSTATES, OP_DTMC_ESTIMATION_DISTRIBUTION,
    OP_DTMC_ESTIMATION_EXPECTED_REWARD, OP_DTMC_ESTIMATION_HITTING_REWARD,
    OP_DTMC_ESTIMATION_HITTING_REWARD_SET, OP_DTMC_ESTIMATION_HITTING_STATE,
    OP_DTMC_ESTIMATION_HITTING_STATE_SET, OP_DTMC_EXECUTION_GRAPH,
    OP_DTMC_HITTING_PROBABILITY, OP_DTMC_HITTING_PROBABILITY_SET,
    OP_DTMC_LIMITING_DISTRIBUTION, OP_DTMC_LIMITING_MATRIX,
    OP_DTMC_LIST_RECURRENT_STATES, OP_DTMC_LIST_STATES,
    OP_DTMC_LIST_TRANSIENT_STATES, OP_DTMC_LONG_RUN_EXPECTED_AVERAGE_REWARD,
    OP_DTMC_LONG_RUN_REWARD, OP_DTMC_MARKOV_TRACE, OP_DTMC_MC_TYPE,
    OP_DTMC_PERIODICITY, OP_DTMC_REWARD_TILL_HIT, OP_DTMC_REWARD_TILL_HIT_SET,
    OP_DTMC_TRANSIENT, OP_DTMC_TRANSIENT_MATRIX, OP_DTMC_TRANSIENT_REWARDS,
    MarkovChainOperations, OperationDescriptions)
from markovchains.utils.statistics import StopConditions
from markovchains.utils.utils import (MarkovChainException, nr_of_steps,
                                      optional_float_or_string_to_string,
                                      pretty_print_matrix, pretty_print_value,
                                      pretty_print_vector,
                                      print_list_of_strings,
                                      print_optional_interval,
                                      print_optional_list,
                                      print_optional_list_of_intervals,
                                      print_sorted_list, print_sorted_set,
                                      print_table, sort_names, stop_criteria,
                                      string_to_float)


def main():
    """Main entry point of the application."""

    # optional help flag explaining usage of each individual operation
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('-oh', '--operationhelp', dest='opHelp', nargs="?", const=" ")
    options, remainder = parser.parse_known_args() # Only use options of parser above

    if options.opHelp: # Check if -oh has been called
        if options.opHelp not in MarkovChainOperations:
            print("Operation '{}' does not exist. List of operations:\n\t- {}".format( \
                options.opHelp, '\n\t- '.join(MarkovChainOperations)))
        else:
            print(f"{options.opHelp}: " \
                  f"{OperationDescriptions[MarkovChainOperations.index(options.opHelp)]}")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Perform operations on discrete-time Markov chains.\n" \
            "https://computationalmodeling.info')
    parser.add_argument('markovchain', help="the Markov chain to analyze")
    parser.add_argument('-op', '--operation', dest='operation', \
                        help="the operation or analysis to perform, one of : {}.\nUse " \
                            "'markovchains -oh OPERATION' for information about the specific "\
                            "operation.".format("; \n".join(MarkovChainOperations)))
    parser.add_argument('-ns', '--numberofsteps', dest='numberOfSteps',
                        help="the number of steps to execute")
    parser.add_argument('-s', '--state', dest='targetState',
                        help="the state for the operation")
    parser.add_argument('-ss', '--stateset', dest='targetStateSet', \
                        help="the set of state for the operation as a non-empty " \
                            "comma-separated list")
    parser.add_argument('-r', '--rewardset', dest='stateRewardSet',\
                        help="the set of state reward for the operation as a non-empty " \
                        "comma-separated list")
    parser.add_argument('-sa', '--startingset', dest='stateStartingSet',
                        help="the set of starting states for the simulation hitting operations " \
                        "as a non-empty comma-separated list")
    parser.add_argument('-c', '--conditions', dest='Conditions',
                        help="The stop conditions for simulating the markovchain [confidence," \
                        "abError,reError,numberOfSteps,numberOfPaths,timeInSeconds]")
    parser.add_argument('-sd', '--seed', dest='Seed',
                        help="Simulation seed for pseudo random variables")

    args = parser.parse_args(remainder)

    if args.operation not in MarkovChainOperations:
        sys.stderr.write(f"Unknown operation: {args.operation}\n")
        sys.exit(1)

    dsl:str = ""

    if args.markovchain:
        try:
            with open(args.markovchain, 'r', encoding='utf-8') as dtmc_file:
                dsl = dtmc_file.read()
        except FileNotFoundError:
            sys.stderr.write(f"File does not exist: {args.markovchain}\n.")
            sys.exit(1)

    try:
        process(args, dsl)
    except MarkovChainException as e:
        sys.stderr.write(f"{e}\n")
        # raise e
        sys.exit(1)

    sys.exit(0)


def require_number_of_steps(args: Any)->int:
    """Ensure number of steps is specified."""
    if args.numberOfSteps is None:
        raise MarkovChainException("number of steps (-ns option) must be specified.")
    try:
        ns: int = nr_of_steps(int(args.numberOfSteps))
    except Exception:
        raise MarkovChainException("Failed to determine number of steps.") # pylint: disable=raise-missing-from
    if ns < 0:
        raise MarkovChainException("Number of steps must be a non-negative number.")
    return ns

def require_target_state(mc: MarkovChain, args: Any)->str:
    """Ensure target state is specified."""
    if args.targetState is None:
        raise MarkovChainException("A target state must be specified with the -s option.")
    s: str = args.targetState
    if not s in mc.states():
        raise MarkovChainException(f"The specified target state {s} does not exist.")
    return args.targetState

def require_target_state_set(mc: MarkovChain, args: Any)->List[str]:
    """Ensure target state set is specified."""
    if args.targetStateSet is None:
        raise MarkovChainException("A target state set must be specified with the -ss option.")
    state_set = [s.strip() for s in args.targetStateSet.split(',')]
    for s in state_set:
        if not s in mc.states():
            raise MarkovChainException(f"State {s} in specified state set does not exist.")
    return state_set

def require_stop_criteria(args: Any)->StopConditions:
    """Ensure stop criteria are specified."""
    if args.Conditions is None:
        raise MarkovChainException("Stop conditions must be specified with the -c option.")
    cc = stop_criteria([string_to_float(i, -1.0) for i in args.Conditions[1:-1].split(',')])
    return cc

def set_seed(args: Any, mc: MarkovChain):
    """Set set for random simulation."""
    if args.Seed is not None:
        mc.setSeed(int(args.Seed))

def set_starting_state_set(args: Any, mc: MarkovChain):
    """Parse starting state set."""
    state_set: List[str]
    if args.stateStartingSet is not None:
        state_set = [s.strip() for s in args.stateStartingSet.split(',')]
    else:
        state_set = mc.states()
    return state_set


def require_markov_chain(mc: Optional[MarkovChain]) -> MarkovChain:
    """Ensure a Markov Chain is specified."""
    if mc is None:
        raise MarkovChainException("A Markov Chain is needed.")
    return mc

def process(args, dsl):
    """Process the command line arguments."""

    operation = args.operation

    mc = None

    if operation in MarkovChainOperations:
        _, mc = MarkovChain.fromDSL(dsl)
        if mc is None:
            sys.exit(1)

    # let the type checker know that we certainly have a Markov Chain from here
    mc = require_markov_chain(mc)

    # just list all states
    if operation == OP_DTMC_LIST_STATES:
        res = mc.states()
        print_sorted_list(res)

    # list the recurrent states
    if operation == OP_DTMC_LIST_RECURRENT_STATES:
        _, recurrent_states = mc.classifyTransientRecurrent()
        print_sorted_list(recurrent_states)

    # list the transient states
    if operation == OP_DTMC_LIST_TRANSIENT_STATES:
        trans, _ = mc.classifyTransientRecurrent()
        print_sorted_list(trans)

    # create graph for a number of steps
    if operation == OP_DTMC_EXECUTION_GRAPH:
        n_steps = require_number_of_steps(args)
        trace = linalg.transpose(mc.executeSteps(n_steps))
        states = mc.states()
        data = {}
        data['k'] = range(0,n_steps+1)
        k = 0
        for s in sort_names(states):
            data[s] = trace[k]
            k += 1
        print(plotSvg(data, states))

    # determine classes of communicating states
    if operation == OP_DTMC_COMMUNICATINGSTATES:
        print("Classes of communicating states:")
        for s in mc.communicatingClasses():
            print_sorted_set(s)

    # classify transient and recurrent states
    if operation == OP_DTMC_CLASSIFY_TRANSIENT_RECURRENT:
        trans, recurrent_states = mc.classifyTransientRecurrent()
        print("Transient states:")
        print_sorted_set(trans)
        print("Recurrent states:")
        print_sorted_set(recurrent_states)

    # classify transient and recurrent states
    if operation == OP_DTMC_PERIODICITY:
        per = mc.classifyPeriodicity()
        print("The set of aperiodic recurrent states is:")
        aper_states =  [s for s in per.keys() if per[s] == 1]
        print_sorted_set(aper_states)

        if len(aper_states) < len(per):
            periodicities = set(per.values())
            if 1 in periodicities:
                periodicities.remove(1)
            for p in periodicities:
                print(f"The set of periodic recurrent states with periodicity {p} is.")
                p_periodic_states =  [s for s in per.keys() if per[s] == p]
                print_sorted_set(p_periodic_states)

    # classify transient and recurrent states
    if operation == OP_DTMC_MC_TYPE:
        mc_type = mc.determineMCType()
        print(f"The type of the MC is: {mc_type}")

    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT:
        n_steps = require_number_of_steps(args)
        trace = mc.executeSteps(n_steps)
        states = mc.states()

        print("Transient analysis:\n")
        print ("State vector:")
        print_list_of_strings(states)

        for k in range(n_steps+1):
            print(f"Step {k}:")
            print("Distribution: ", end="")
            pretty_print_vector(trace[k])

    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT_REWARDS:
        n_steps = require_number_of_steps(args)
        trace = mc.executeSteps(n_steps)

        print("Transient reward analysis:")
        for k in range(n_steps+1):
            print(f"\nStep {k}:")
            print("Expected Reward: ", end='')
            pretty_print_value(mc.rewardForDistribution(trace[k]))


    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT_MATRIX:
        n_steps = require_number_of_steps(args)
        mat = mc.transitionMatrix()

        print ("State vector:")
        print_list_of_strings(mc.states())
        print("Transient analysis:\n")
        print(f"Matrix for {n_steps} steps:\n")
        pretty_print_matrix(matPower(mat, n_steps))

    if operation == OP_DTMC_LIMITING_MATRIX:
        mat = mc.limitingMatrix()
        print ("State vector:")
        print_list_of_strings(mc.states())
        print ("Limiting Matrix:\n")
        pretty_print_matrix(mat)

    if operation == OP_DTMC_LIMITING_DISTRIBUTION:
        l_dist: TVector = mc.limitingDistribution()

        print ("State vector:")
        print_list_of_strings(mc.states())
        print ("Limiting Distribution:")
        pretty_print_vector(l_dist)

    if operation == OP_DTMC_LONG_RUN_REWARD:
        mc_type = mc.determineMCType()
        r = mc.longRunReward()
        if 'non-ergodic' in mc_type:
            print(f"The long-run expected average reward is: {r}\n")
        else:
            print(f"The long-run expected reward is: {r}\n")

    if operation == OP_DTMC_HITTING_PROBABILITY:
        s = require_target_state(mc, args)
        prob = mc.hittingProbabilities(s)
        print(f"The hitting probabilities for {s} are:")
        for t in sort_names(mc.states()):
            print(f"f({t}, {s}) = {prob[t]}")

    if operation == OP_DTMC_REWARD_TILL_HIT:
        s = require_target_state(mc, args)
        res = mc.rewardTillHit(s)
        print(f"The expected rewards until hitting {s} are:")
        for s in sort_names(res.keys()):
            print(f"From state {s}: {res[s]}")

    if operation == OP_DTMC_HITTING_PROBABILITY_SET:
        target_state_set = require_target_state_set(mc, args)
        prob = mc.hittingProbabilitiesSet(target_state_set)
        print(f"The hitting probabilities for {{{', '.join(prob)}}} are:")
        ss = ', '.join(target_state_set)
        for t in sort_names(mc.states()):
            print(f"f({t}, {{{ss}}}) = {prob[t]}")

    if operation == OP_DTMC_REWARD_TILL_HIT_SET:
        s = require_target_state_set(mc, args)
        res = mc.rewardTillHitSet(s)
        print(f"The expected rewards until hitting {{{', '.join(s)}}} are:")
        for t in sort_names(res.keys()):
            print(f"From state {t}: {res[t]}")

    if operation == OP_DTMC_MARKOV_TRACE:
        set_seed(args, mc)
        n_steps = require_number_of_steps(args)
        trace = mc.markovTrace(n_steps)
        print(f"{trace}")

    if operation == OP_DTMC_LONG_RUN_EXPECTED_AVERAGE_REWARD:
        set_seed(args, mc)
        if args.targetState:
            mc.setRecurrentState(args.targetState)
        crit = require_stop_criteria(args)
        statistics, stop = mc.longRunExpectedAverageReward(crit)
        if statistics.cycle_count() == 0:
            print("Recurrent state has not been reached, no realizations found")
        else:
            print(f"Simulation termination reason: {stop}")
            print("The long run expected average reward is:")
            print("\tEstimated mean: " \
                f"{optional_float_or_string_to_string(statistics.mean_estimate_result())}")
            print("\tConfidence interval: " \
                f"{print_optional_interval(statistics.confidence_interval())}")
            print("\tAbsolute error bound: " \
                f"{optional_float_or_string_to_string(statistics.ab_error())}")
            print("\tRelative error bound: " \
                f"{optional_float_or_string_to_string(statistics.re_error())}")
            print(f"\tNumber of cycles: {statistics.cycle_count()}")

    if operation == OP_DTMC_CEZARO_LIMIT_DISTRIBUTION:
        set_seed(args, mc)
        if args.targetState:
            mc.setRecurrentState(args.targetState)
        crit = require_stop_criteria(args)
        distribution_statistics, stop = mc.cezaroLimitDistribution(crit)

        if distribution_statistics is None:
            print("Recurrent state has not been reached, no realizations found")
        else:
            print(f"Simulation termination reason: {stop}")
            cld = print_optional_list(distribution_statistics.point_estimates(), \
                                      'Could not be determined')
            print(f"Cezaro limit distribution: {cld}")
            print(f"Number of cycles: {distribution_statistics.cycle_count()}\n")
            dist: Optional[List[float]] = distribution_statistics.point_estimates()
            if dist is not None:
                intervals = distribution_statistics.confidence_intervals()
                ab_error = distribution_statistics.ab_error()
                re_error = distribution_statistics.re_error()
                states: List[str] = mc.states()
                for i, s in enumerate(states):
                    print(f"[{s}]: {dist[i]:.4f}")
                    ci = print_optional_interval(None if intervals is None else intervals[i])
                    print(f"\tConfidence interval: {ci}")
                    aeb = optional_float_or_string_to_string(ab_error[i])
                    print(f"\tAbsolute error bound: {aeb}")
                    reb = optional_float_or_string_to_string(re_error[i])
                    print(f"\tRelative error bound: {reb}")
                    print("\n")

    if operation == OP_DTMC_ESTIMATION_EXPECTED_REWARD:
        set_seed(args, mc)
        n_steps = require_number_of_steps(args)
        crit = require_stop_criteria(args)
        statistics, stop = mc.estimationExpectedReward(crit, n_steps)
        print(f"Simulation termination reason: {stop}")
        er = optional_float_or_string_to_string(statistics.mean_estimate_result())
        print(f"\tExpected reward: {er}")
        ci = print_optional_interval(statistics.confidence_interval())
        print(f"\tConfidence interval: {ci}")
        aeb = optional_float_or_string_to_string(statistics.ab_error())
        print(f"\tAbsolute error bound: {aeb}")
        reb = optional_float_or_string_to_string(statistics.re_error())
        print(f"\tRelative error bound: {reb}")
        print("\tNumber of realizations: ", statistics.cycle_count())

    if operation == OP_DTMC_ESTIMATION_DISTRIBUTION:
        set_seed(args, mc)
        states = mc.states()
        n_steps = require_number_of_steps(args)
        crit = require_stop_criteria(args)
        distribution_statistics, stop = mc.estimationTransientDistribution(crit, n_steps)
        print(f"Simulation termination reason: {stop}")
        print(f"The estimated distribution after {n_steps} steps of " \
              f"[{', '.join(states)}] is as follows:")
        print(f"\tDistribution: {print_optional_list(distribution_statistics.point_estimates())}")
        ci = print_optional_list_of_intervals(distribution_statistics.confidence_intervals())
        print(f"\tConfidence intervals: {ci}")
        aeb = optional_float_or_string_to_string(distribution_statistics.max_ab_error())
        print(f"\tAbsolute error bound: {aeb}")
        reb = optional_float_or_string_to_string(distribution_statistics.max_re_error())
        print(f"\tRelative error bound: {reb}")
        print("\tNumber of realizations: ", distribution_statistics.cycle_count())

    if operation == OP_DTMC_ESTIMATION_HITTING_STATE:
        set_seed(args, mc)
        state_set = set_starting_state_set(args, mc)
        s = require_target_state(mc, args)
        crit = require_stop_criteria(args)
        statistics_dict, stop = mc.estimationHittingProbabilityState(crit, s, state_set)
        if statistics_dict is None:
            print("A timeout has occurred during the analysis.")
        else:
            d_stop: Dict[str,str] = stop  # type: ignore
            print(f"Estimated hitting probabilities for {s} are:")
            table_hs: List[Union[str,List[str]]] = []
            for i, t in enumerate(state_set):
                statistics = statistics_dict[t]
                table_hs.append([
                    f"f({t}, {s}) = " \
                        f"{optional_float_or_string_to_string(statistics.mean_estimate_result())}",
                    f"int: {print_optional_interval(statistics.confidence_interval())}",
                    f"abEr: {optional_float_or_string_to_string(statistics.ab_error())}",
                    f"reEr: {optional_float_or_string_to_string(statistics.re_error())}",
                    f"#paths: {statistics.nr_paths()}",
                    f"stop: {d_stop[t]}"
                ])
            print_table(table_hs, 4)

    if operation == OP_DTMC_ESTIMATION_HITTING_REWARD:
        set_seed(args, mc)
        state_set = set_starting_state_set(args, mc)
        s = require_target_state(mc, args)
        crit = require_stop_criteria(args)
        statistics_dict, stop = mc.estimationRewardUntilHittingState(crit, s, state_set)
        if statistics_dict is None:
            print("A timeout has occurred during the analysis.")
        else:
            print(f"Estimated cumulative reward until hitting {s} are:")
            table_rs: List[Union[str,List[str]]] = []
            for i, t in enumerate(state_set):
                statistics = statistics_dict[t]
                if not isinstance(statistics.mean_estimate_result(), float):
                    table_rs.append(f"From state {state_set[i]}: " \
                        f"{optional_float_or_string_to_string(statistics.mean_estimate_result())}")
                else:
                    d_stop: Dict[str,str] = stop  # type: ignore
                    ss = optional_float_or_string_to_string(statistics.mean_estimate_result())
                    table_rs.append([
                        f"From state {state_set[i]}: {ss}",
                        f"int: {print_optional_interval(statistics.confidence_interval())}",
                        f"abEr: {optional_float_or_string_to_string(statistics.ab_error())}",
                        f"reEr: {optional_float_or_string_to_string(statistics.re_error())}",
                        f"#paths: {statistics.nr_paths()}",
                        f"stop: {d_stop[t]}"
                    ])
            print_table(table_rs, 4)

    if operation == OP_DTMC_ESTIMATION_HITTING_STATE_SET:
        set_seed(args, mc)
        state_set = set_starting_state_set(args, mc)
        s = require_target_state_set(mc, args)
        crit = require_stop_criteria(args)
        statistics_dict, stop = mc.estimationHittingProbabilityStateSet(crit, s, state_set)
        if statistics_dict is None:
            print("A timeout has occurred during the analysis.")
        else:
            print(f"Estimated hitting probabilities for {{{', '.join(s)}}} are:")
            table: List[Union[str,List[str]]] = []
            for i, t in enumerate(state_set):
                statistics = statistics_dict[t]
                if not isinstance(statistics.mean_estimate_result(), float):
                    table.append(f"From state {state_set[i]}: {statistics.mean_estimate_result()}")
                else:
                    d_stop: Dict[str,str] = stop  # type: ignore
                    mer = optional_float_or_string_to_string(statistics.mean_estimate_result())
                    table.append([
                        f"f({state_set[i]}, {{{', '.join(s)}}}) = {mer}",
                        f"int: {print_optional_interval(statistics.confidence_interval())}",
                        f"abEr: {optional_float_or_string_to_string(statistics.ab_error())}",
                        f"reEr: {optional_float_or_string_to_string(statistics.re_error())}",
                        f"#paths: {statistics.nr_paths()}",
                        f"stop: {d_stop[t]}"
                    ])
            print_table(table, 4)


    if operation == OP_DTMC_ESTIMATION_HITTING_REWARD_SET:
        set_seed(args, mc)
        state_set = set_starting_state_set(args, mc)
        s = require_target_state_set(mc, args)
        crit = require_stop_criteria(args)
        statistics_dict, stop = mc.estimationRewardUntilHittingStateSet(crit, s, state_set)
        if statistics_dict is None:
            print("A timeout has occurred during the analysis.")
        else:
            print(f"Estimated cumulative reward until hitting {{{', '.join(s)}}} are:")
            table_r_set: List[Union[str,List[str]]] = []
            for i, t in enumerate(state_set):
                statistics = statistics_dict[t]
                if not isinstance(statistics.mean_estimate_result(), float):
                    table_r_set.append(f"From state {state_set[i]}: " \
                        f"{optional_float_or_string_to_string(statistics.mean_estimate_result())}")
                else:
                    d_stop: Dict[str,str] = stop  # type: ignore
                    mer = optional_float_or_string_to_string(statistics.mean_estimate_result())
                    table_r_set.append([
                        f"From state {state_set[i]}: {mer}",
                        f"int: {print_optional_interval(statistics.confidence_interval())}",
                        f"abEr: {optional_float_or_string_to_string(statistics.ab_error())}",
                        f"reEr: {optional_float_or_string_to_string(statistics.re_error())}",
                        f"#paths: {statistics.nr_paths()}",
                        f"stop: {d_stop[t]}"
                    ])
            print_table(table_r_set, 4)

if __name__ == "__main__":
    main()
