
'''Operations on Markov chains '''

import argparse

from markovchains.libdtmc import MarkovChain
from markovchains.utils.graphs import plotSvg
from markovchains.utils.linalgebra import matPower, printMatrix
from markovchains.utils.utils import sortNames, printList, printDList, print4F, stringToFloat, stopCriteria, nrOfSteps, printSortedList, printSortedSet, printVector
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
                        help="the operation or analysis to perform, use 'markovchains -oh OPERATION' for information about the specific operation ")
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

    if args.markovchain:
        try:
            with open(args.markovchain, 'r') as dtmcFile:
                dsl = dtmcFile.read()
        except FileNotFoundError as e:
            sys.stderr.write("File does not exist: {}\n.".format(args.markovchain))
            exit(1)

    try:
        process(args, dsl)
    except Exception as e:
        sys.stderr.write("An error occurred: {}\n".format(e))
        # raise e
        exit(1)

    exit(0)



def requireNumberOfSteps(args):
    if args.numberOfSteps is None:
        raise Exception("numberOfSteps must be specified.")
    try:
        return nrOfSteps(int(args.numberOfSteps))
    except Exception as e:
        sys.stderr.write("Failed to determine number of steps.\n")

def requireTargetState(args):
    if args.targetState is None:
        raise Exception("A target state must be specified.")
    return args.targetState

def requireTargetStateSet(args):
    if args.targetStateSet is None:
        raise Exception("A target state set must be specified.")
    return [s.strip() for s in args.targetStateSet.split(',')]

def requireStopCriteria(args):
    if args.Conditions is None:
        raise Exception("Stop conditions must be specified.")
    return stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])

def setSeed(args, M):
    if args.Seed is not None:
        M.setSeed(int(args.Seed))

def setStartingStateSet(args, M):
    if args.stateStartingSet is not None:
        S = [s.strip() for s in args.stateStartingSet.split(',')]
    else:
        S = M.states()
    return S


def process(args, dsl):

    operation = args.operation

    if operation in MarkovChainOperations:
        name, M = MarkovChain.fromDSL(dsl)
        if M is None:
            exit(1)

    # just list all states
    if operation == OP_DTMC_LIST_STATES:
        res = M.states()
        printSortedList(res)

    # list the recurrent states
    if operation == OP_DTMC_LIST_RECURRENT_STATES:
        _, recurr = M.classifyTransientRecurrent()
        printSortedList(recurr)

    # list the transient states
    if operation == OP_DTMC_LIST_TRANSIENT_STATES:
        trans, _ = M.classifyTransientRecurrent()
        printSortedList(trans)

    # create graph for a number of steps
    if operation == OP_DTMC_EXECUTION_GRAPH:
        N = requireNumberOfSteps(args)
        trace = M.executeSteps(N)
        states = M.states()
        data = dict()
        data['k'] = range(0,N+1)
        k = 0
        for s in sortNames(states):
            data[s] = trace[:,k]
            k += 1
        print(plotSvg(data, states))

    # determine classes of communicating states
    if operation == OP_DTMC_COMMUNICATINGSTATES:
        print("Classes of communicating states:")
        for s in M.communicatingClasses():
            printSortedSet(s)

    # classify transient and recurrent states
    if operation == OP_DTMC_CLASSIFY_TRANSIENT_RECURRENT:
        trans, recurr = M.classifyTransientRecurrent()
        print("Transient states:")
        printSortedSet(trans)
        print("Recurrent states:")
        printSortedSet(recurr)

    # classify transient and recurrent states
    if operation == OP_DTMC_PERIODICITY:
        per = M.classifyPeriodicity()
        print("The set of aperiodic recurrent states is:")
        aperStates =  [s for s in per.keys() if per[s] == 1]
        printSortedSet(aperStates)

        if len(aperStates) < len(per):
            periodicities = set(per.values())
            if 1 in periodicities:
                periodicities.remove(1)
            for p in periodicities:
                print("The set of periodic recurrent states with periodicity {} is.".format(p))
                pperStates =  [s for s in per.keys() if per[s] == p]
                printSortedSet(pperStates)

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
        printVector(states)

        for k in range(N+1):
            print("Step {}:".format(k))
            print("Distribution: " +  printList(trace[k,:]) + "\n")

    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT_REWARDS:
        N = requireNumberOfSteps(args)
        trace = M.executeSteps(N)

        print("Transient reward analysis:\n")
        for k in range(N+1):
            print("Step {}:".format(k))
            print("Expected Reward: {:.4f}\n".format(M.rewardForDistribution(trace[k,:])))

    # determine transient behavior for a number of steps
    if operation == OP_DTMC_TRANSIENT_MATRIX:
        N = requireNumberOfSteps(args)
        mat = M.transitionMatrix()

        print ("State vector:")
        printVector(M.states())
        print("Transient analysis:\n")
        print ("Matrix for {} steps:\n".format(N))
        printMatrix(matPower(mat, N))

    if operation == OP_DTMC_LIMITING_MATRIX:
        mat = M.limitingMatrix()
        print ("State vector:")
        printVector(M.states())
        print ("Limiting Matrix:\n")
        printMatrix(mat)

    if operation == OP_DTMC_LIMITING_DISTRIBUTION:
        dist = M.limitingDistribution()

        print ("State vector:")
        printVector(M.states())
        print ("Limiting Distribution:")
        print("{}\n".format(printList(dist)))

    if operation == OP_DTMC_LONG_RUN_REWARD:
        mcType = M.determineMCType()
        r = M.longRunReward()
        if 'non-ergodic' in mcType:
            print("The long-run expected average reward is: {:.4f}\n".format(r))
        else:
            print("The long-run expected reward is: {:.4f}\n".format(r))

    if operation == OP_DTMC_HITTING_PROBABILITY:
        s = requireTargetState(args)
        prob = M.hittingProbabilities(s)
        print("The hitting probabilities for {} are:".format(s))
        for t in sortNames(M.states()):
            print("f({}, {}) = {:.4f}".format(t, s, prob[t]))

    if operation == OP_DTMC_REWARD_TILL_HIT:
        s = requireTargetState(args)
        res = M.rewardTillHit(s)
        print("The expected rewards until hitting {} are:".format(s))
        for s in sortNames(res.keys()):
            print("From state {}: {:.4f}".format(s, res[s]))

    if operation == OP_DTMC_HITTING_PROBABILITY_SET:
        targetStateSet = requireTargetStateSet(args)
        prob = M.hittingProbabilitiesSet(targetStateSet)
        print("The hitting probabilities for {{{}}} are:".format(', '.join(prob)))
        ss = ', '.join(targetStateSet)
        for t in sortNames(M.states()):
            print("f({}, {{{}}}) = {:.4f}".format(t, ss, prob[t]))

    if operation == OP_DTMC_REWARD_TILL_HIT_SET:
        s = requireTargetStateSet(args)
        res = M.rewardTillHitSet(s)
        print("The expected rewards until hitting {{{}}} are:".format(', '.join(s)))
        for t in sortNames(res.keys()):
            print("From state {}: {:.4f}".format(t, res[t]))
    
    if operation == OP_DTMC_MARKOV_TRACE:
        setSeed(args, M)
        N = requireNumberOfSteps(args)
        trace = M.markovTrace(N)
        print("{}".format(trace))

    if operation == OP_DTMC_LONG_RUN_EXPECTED_AVERAGE_REWARD:
        setSeed(args, M)
        M.setRecurrentState(args.targetState) # targetState is allowed to be None
        C = requireStopCriteria(args)
        interval, abError, reError, esMean, n, stop = M.longRunExpectedAverageReward(C)
        if any(i==None for i in interval):
            print("Recurrent state has not been reached, no realizations found")
        else:
            print("Simulation termination reason: {}".format(stop))
            print("The long run expected average reward is:")
            print("\tEstimated mean: {}".format(print4F(esMean)))
            print("\tConfidence interval: {}".format(printList(interval)))
            print("\tAbsolute error bound: {}".format(print4F(abError)))
            print("\tRelative error bound: {}".format(print4F(reError)))
            print("\tNumber of cycles: {}".format(n))

    if operation == OP_DTMC_CEZARO_LIMIT_DISTRIBUTION:
        setSeed(args, M)
        M.setRecurrentState(args.targetState) # targetState is allowed to be None
        C = requireStopCriteria(args)
        limit, interval, abError, reError, n, stop = M.cezaroLimitDistribution(C)
        if limit is None:
            print("Recurrent state has not been reached, no realisations found")
        else:
            print("Simulation termination reason: {}".format(stop))
            print("Cezaro limit distribution: {}".format(printList(limit)))
            print("Number of cycles: {}\n".format(n))
            for i, l in enumerate(limit):
                print("[{}]: {:.4f}".format(i, l))
                print("\tConfidence interval: {}".format(printList(interval[i])))
                print("\tAbsolute error bound: {}".format(print4F(abError[i])))
                print("\tRelative error bound: {}".format(print4F(reError[i])))
                print("\n")

    if operation == OP_DTMC_ESTIMATION_EXPECTED_REWARD:
        setSeed(args, M)
        N = requireNumberOfSteps(args)
        C = requireStopCriteria(args)
        u, interval, abError, reError, nr_of_paths, stop = M.estimationExpectedReward(C, N)
        print("Simulation termination reason: {}".format(stop))
        print("\tExpected reward: {:.4f}".format(u))
        print("\tConfidence interval: {}".format(printList(interval)))
        print("\tAbsolute error bound: {}".format(print4F(abError)))
        print("\tRelative error bound: {}".format(print4F(reError)))
        print("\tNumber of paths: ", nr_of_paths)

    if operation == OP_DTMC_ESTIMATION_DISTRIBUTION:
        setSeed(args, M)
        states = M.states()
        N = requireNumberOfSteps(args)
        C = requireStopCriteria(args)
        distribution, intervals, abError, reError, nr_of_paths, stop = M.estimationDistribution(C, N)
        print("Simulation termination reason: {}".format(stop))
        print("The estimated distribution after {} steps of [{}] is as follows:".format(N, ", ".join(states)))
        print("\tDistribution: {}".format(printList(distribution)))
        print("\tConfidence intervals: {}".format(printDList(intervals)))
        print("\tAbsolute error bound: {}".format(print4F(abError)))
        print("\tRelative error bound: {}".format(print4F(reError)))
        print("\tNumber of paths: ", nr_of_paths)

    if operation == OP_DTMC_ESTIMATION_HITTING_STATE:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetState(args)
        C = requireStopCriteria(args)
        hitting_probability,nr_of_paths,abErrors,reErrors,intervals,stop = M.estimationHittingState(C, s, True, True, S)
        print("Estimated hitting probabilities for {} are:".format(s))
        for i in range(len(hitting_probability)):
            print("f({}, {}) = {}\tint:{}\tabEr:{}\treEr:{}\t#paths:{}\tstop:{}".format(
                S[i], s, print4F(hitting_probability[i]), printList(intervals[i]), 
                print4F(abErrors[i]), print4F(reErrors[i]), nr_of_paths[i], stop[i]
            ))
                
    if operation == OP_DTMC_ESTIMATION_HITTING_REWARD:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetState(args)
        C = requireStopCriteria(args)
        cumulative_reward,nr_of_paths,abErrors,reErrors,intervals,stop = M.estimationHittingState(C, s, True, False, S)
        print("Estimated cumulative reward until hitting {} are:".format(s))
        for i in range(len(cumulative_reward)):
            if type(cumulative_reward[i]) is str:
                print("From state {}: {}".format(S[i], cumulative_reward[i]))
            else:
                print("From state {}: {}\tint:{}\tabEr:{}\treEr:{}\t#paths:{}\tstop:{}".format(
                    S[i], print4F(cumulative_reward[i]), printList(intervals[i]), 
                    print4F(abErrors[i]), print4F(reErrors[i]), nr_of_paths[i], stop[i]
                ))
    
    if operation == OP_DTMC_ESTIMATION_HITTING_STATE_SET:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetStateSet(args)
        C = requireStopCriteria(args)
        hitting_probability,nr_of_paths,abErrors,reErrors,intervals,stop = M.estimationHittingState(C, s, False, True, S)
        print("Estimated hitting probabilities for {{{}}} are:".format(', '.join(s)))
        for i in range(len(hitting_probability)):
            print("f({}, {{{}}}) = {}\tint:{}\tabEr:{}\treEr:{}\t#paths:{}\tstop:{}".format(
                S[i], ', '.join(s), print4F(hitting_probability[i]),
                printList(intervals[i]), print4F(abErrors[i]), print4F(reErrors[i]), nr_of_paths[i], stop[i]
            ))

    if operation == OP_DTMC_ESTIMATION_HITTING_REWARD_SET:
        setSeed(args, M)
        S = setStartingStateSet(args, M)
        s = requireTargetStateSet(args)
        C = requireStopCriteria(args)
        cumulative_reward,nr_of_paths,abErrors,reErrors,intervals,stop = M.estimationHittingState(C, s, False, False, S)
        print("Estimated cumulative reward until hitting {{{}}} are:".format(', '.join(s)))
        for i in range(len(cumulative_reward)):
            if type(cumulative_reward[i]) is str:
                print("From state {}: {}".format(S[i], cumulative_reward[i]))
            else:
                print("From state {}: {}\tint:{}\tabEr:{}\treEr:{}\t#paths:{}\tstop:{}".format(
                    S[i], print4F(cumulative_reward[i]), printList(intervals[i]), 
                    print4F(abErrors[i]), print4F(reErrors[i]), nr_of_paths[i], stop[i]
                ))
                
if __name__ == "__main__":
    main()