
'''Operations on Markov chains '''

import argparse
from markovchains.libdtmc import MarkovChain
from markovchains.utils.graphs import plotSvg
from markovchains.utils.linalgebra import matPower, printMatrix
from markovchains.utils.utils import sortNames, printList, printDList, print4F, stringToFloat, stopCriteria, nrOfSteps
import sys
import nose

MarkovChainOperations = [
    "liststates",
    "listrecurrentstates",
    "listtransientstates",
    "communicatingstates",
    "classifytransientrecurrent",
    "hittingprobability",
    "hittingprobabilityset",
    "rewardtillhit",
    "rewardtillhitset",
    "periodicity",
    "mctype",
    "transient",
    "transientRewards",
    "transientMatrix",
    "limitingMatrix",
    "limitingDistribution",
    "longRunReward",
    "executiongraph",
    "markovtrace",
    "longrunexpectedaveragereward",
    "cezarolimitdistribution",
    "estimationexpectedreward",
    "estimationdistribution",
    "estimationhittingstate",
    "estimationhittingreward",
    "estimationhittingstateset",
    "estimationhittingrewardset"
]

def main():

    operationDescriptions = [
        "List all states of the markov chain\n\tNo flags",
        "List all recurrent states of the markov chain\n\tNo flags",
        "List all transient states of the markov chain\n\tNo flags",
        "Provides list of communicating state sets\n\tNo flags",
        "Lists the transient and recurrent states\n\tNo flags",
        "Provides the hitting probability for a specified state\n\trequired flag:\n\t\t[-s, --state]: Target state",
        "Provides the hitting probability for a specified state set\n\trequired flag:\n\t\t[-ss, --stateset]: Set of target states (comma seperated)",
        "Expected reward untill hitting specified state\n\trequired flag:\n\t\t[-s, --state]: Target state",
        "Expected reward untill hitting specified state set\n\trequired flag:\n\t\t[-ss, --stateset]: Set of target states (comma seperated)",
        "Lists aperiodic and periodic state states\n\tNo flags",
        "Provides type of markov chain: (non-)ergodic (non-)unichain\n\tNo flags",
        "Transient analysis for specified number of steps\n\trequired flag:\n\t\t[-ns, --numberofsteps]: Number of steps",
        "Transient analysis of reward afeter specified number of steps\n\trequired flag:\n\t\t[-ns, --numberofsteps]: Number of steps",
        "Transient matrix for specified number of steps\n\trequired flag:\n\t\t[-ns, --numberofsteps]: Number of steps",
        "Provides limiting Matrix\n\tNo flags",
        "Provides limiting Distribution\n\tNo flags",
        "Long-run expected average reward\n\tNo flags",
        "Prints execution graphs xml file for specified number of steps\n\trequired flag:\n\t\t[-ns, --numberofsteps]: Number of steps",
        '''Provides simulation trace through markov chain
        required flag:\n\t\t[-ns, --numberofsteps]: Number of steps
        Optional flag:\n\t\t[-sd, --seed]: SEED''',
        '''Long run expected average reward through simulation
        required flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions
        Optional flags:\n\t\t[-sd, --seed]: Seed\n\t\t[-s, --state]: Recurrent state''',
        '''Cezarolimit distribution through simulation\n\trequired flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions
        Optional flags:\n\t\t[-sd, --seed]: Seed\n\t\t[-s, --state]: Recurrent state''',
        '''Estimation of exected reward by simulation
        required flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions\n\t\t[-ns, --numberofsteps]: Number of steps
        Optional flag:\n\t\t[-sd, --seed]: Seed''',
        '''Estimation of distribution by simulation after specified number of steps
        required flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions\n\t\t[-ns, --numberofsteps]: Number of steps
        Optional flag:\n\t\t[-sd, --seed]: Seed''',
        '''Estimation of hitting state probabilites by simulation
        required flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions\n\t\t[-s, --state]: Target state
        Optional flag:\n\t\t[-sd, --seed]: Seed\n\t\t[-sa, --startingset]: Set of starting states to simulate''',
        '''Estimation of cumulative reward hitting state by simulation
        required flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions\n\t\t[-s, --state]: Target state
        Optional flag:\n\t\t[-sd, --seed]: Seed\n\t\t[-sa, --startingset]: Set of starting states to simulate''',
        '''Estimation of hitting state set probabilites by simulation
        required flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions\n\t\t[-ss, --stateset]: Set of target states (comma seperated)
        Optional flag:\n\t\t[-sd, --seed]: Seed\n\t\t[-sa, --startingset]: Set of starting states to simulate''',
        '''Estimation of cumulative reward hitting state set probabilites by simulation
        required flag:\n\t\t[-c, --conditions]: Simulation (Stop) conditions\n\t\t[-ss, --stateset]: Set of target states (comma seperated)
        Optional flag:\n\t\t[-sd, --seed]: Seed\n\t\t[-sa, --startingset]: Set of starting states to simulate'''
    ]

    # optional help flag explaining usage of each individual operation
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('-oh', '--operationhelp', dest='opHelp', nargs="?", const=" ")
    options, remainder = parser.parse_known_args() # Only use options of parser above

    if options.opHelp: # Check if -oh has been called
        if options.opHelp not in MarkovChainOperations:
            print("Operation '{}' does not exist. List of operations:\n\t- {}".format(options.opHelp, "\n\t- ".join(MarkovChainOperations)))
        else:
            print("{}: {}".format(options.opHelp, operationDescriptions[MarkovChainOperations.index(options.opHelp)]))        
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



def process(args, dsl):

    if args.operation in MarkovChainOperations:
        name, M = MarkovChain.fromDSL(dsl)
        if M is None:
            exit(1)

    # just list all states
    if args.operation == "liststates":
        res = M.states()
        print("{}".format(", ".join(sortNames(res))))

    # list the recurrent states
    if args.operation == "listrecurrentstates":
        _, recurr = M.classifyTransientRecurrent()
        print("{}".format(", ".join(sortNames(recurr))))

    # list the transient states
    if args.operation == "listtransientstates":
        trans, _ = M.classifyTransientRecurrent()
        print("{}".format(", ".join(sortNames(trans))))

    # function does not exist inside class MarkovChain
    # compute equilibrium distribution
    # if args.operation == "equilibrium":
    #     res = M.equilibrium()
    #     print(res)

    # create graph for a number of steps
    if args.operation == "executiongraph":
        N = int(args.numberOfSteps)
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
    if args.operation == "communicatingstates":
        print("Classes of communicating states:")
        for s in M.communicatingClasses():
            print("{{{}}}".format(", ".join(sortNames(s))))

    # classify transient and recurrent states
    if args.operation == "classifytransientrecurrent":
        trans, recurr = M.classifyTransientRecurrent()
        print("Transient states:")
        print("{{{}}}".format(", ".join(sortNames(trans))))
        print("Recurrent states:")
        print("{{{}}}".format(", ".join(sortNames(recurr))))

    # classify transient and recurrent states
    if args.operation == "periodicity":
        per = M.classifyPeriodicity()
        print("The set of aperiodic recurrent states is:")
        aperStates =  [s for s in per.keys() if per[s] == 1]
        print("{{{}}}".format(", ".join(sortNames(aperStates))))

        if len(aperStates) < len(per):
            periodicities = set(per.values())
            if 1 in periodicities:
                periodicities.remove(1)
            for p in periodicities:
                print("The set of periodic recurrent states with periodicity {} is.".format(p))
                pperStates =  [s for s in per.keys() if per[s] == p]
                print("{{{}}}".format(", ".join(sortNames(pperStates))))

    # classify transient and recurrent states
    if args.operation == "mctype":
        mcType = M.determineMCType()
        print("The type of the MC is: {}".format(mcType))

    # determine transient behavior for a number of steps
    if args.operation == "transient":
        N = int(args.numberOfSteps)
        trace = M.executeSteps(N)
        states = M.states()

        print("Transient analysis:\n")
        print ("State vector:")
        print ("[{}]\n".format(", ".join(states)))

        for k in range(N+1):
            print("Step {}:".format(k))
            print("Distribution: " +  printList(trace[k,:]) + "\n")

    # determine transient behavior for a number of steps
    if args.operation == "transientRewards":
        N = int(args.numberOfSteps)
        trace = M.executeSteps(N)

        print("Transient reward analysis:\n")
        for k in range(N+1):
            print("Step {}:".format(k))
            print("Expected Reward: {:.4f}\n".format(M.rewardForDistribution(trace[k,:])))

    # determine transient behavior for a number of steps
    if args.operation == "transientMatrix":
        N = int(args.numberOfSteps)
        mat = M.transitionMatrix()

        print ("State vector:")
        print ("[{}]\n".format(", ".join(M.states())))
        print("Transient analysis:\n")
        print ("Matrix for {} steps:\n".format(N))
        printMatrix(matPower(mat, N))

    if args.operation == "limitingMatrix":
        mat = M.limitingMatrix()
        print ("State vector:")
        print ("[{}]\n".format(", ".join(M.states())))
        print ("Limiting Matrix:\n")
        printMatrix(mat)

    if args.operation == "limitingDistribution":
        dist = M.limitingDistribution()

        print ("State vector:")
        print ("[{}]\n".format(", ".join(M.states())))
        print ("Limiting Distribution:")
        print("{}\n".format(printList(dist)))

    if args.operation == "longRunReward":
        mcType = M.determineMCType()
        r = M.longRunReward()
        if 'non-ergodic' in mcType:
            print("The long-run expected average reward is: {:.4f}\n".format(r))
        else:
            print("The long-run expected reward is: {:.4f}\n".format(r))


    if args.operation == "hittingprobability":
        s = args.targetState
        prob = M.hittingProbabilities(s)
        print("The hitting probabilities for {} are:".format(s))
        for t in sortNames(M.states()):
            print("f({}, {}) = {:.4f}".format(t, s, prob[t]))

    if args.operation == "rewardtillhit":
        s = args.targetState
        res = M.rewardTillHit(s)
        print("The expected rewards until hitting {} are:".format(s))
        for s in sortNames(res.keys()):
            print("From state {}: {:.4f}".format(s, res[s]))

    if args.operation == "hittingprobabilityset":
        s = [s.strip() for s in args.targetStateSet.split(',')]
        prob = M.hittingProbabilitiesSet(s)
        print("The hitting probabilities for {{{}}} are:".format(', '.join(s)))
        ss = ', '.join(s)
        for t in sortNames(M.states()):
            print("f({}, {{{}}}) = {:.4f}".format(t, ss, prob[t]))

    if args.operation == "rewardtillhitset":
        s = [s.strip() for s in args.targetStateSet.split(',')]
        res = M.rewardTillHitSet(s)
        print("The expected rewards until hitting {{{}}} are:".format(', '.join(s)))
        for t in sortNames(res.keys()):
            print("From state {}: {:.4f}".format(t, res[t]))
    
    if args.operation == "markovtrace":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        N = int(args.numberOfSteps)
        trace = M.markovTrace(N)
        print("{}".format(trace))

    if args.operation == "longrunexpectedaveragereward":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        M.setRecurrentState(args.targetState)
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
        interval, abError, reError, esMean, n, stop = M.longRunExpectedAverageReward(C)
        if any(i==None for i in interval):
            print("Recurrent state has not been reached, no realisations found")
        else:
            print("Simulation termination reason: {}".format(stop))
            print("The long run expected average reward is:")
            print("\tEstimated mean: {}".format(print4F(esMean)))
            print("\tConfidence interval: {}".format(printList(interval)))
            print("\tAbsolute error bound: {}".format(print4F(abError)))
            print("\tRelative error bound: {}".format(print4F(reError)))
            print("\tNumber of cycles: {}".format(n))

    if args.operation == "cezarolimitdistribution":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        M.setRecurrentState(args.targetState)
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
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

    if args.operation == "estimationexpectedreward":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        N = int(nrOfSteps(args.numberOfSteps))
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
        u, interval, abError, reError, nr_of_paths, stop = M.estimationExpectedReward(C, N)
        print("Simulation termination reason: {}".format(stop))
        print("\tExpected reward: {:.4f}".format(u))
        print("\tConfidence interval: {}".format(printList(interval)))
        print("\tAbsolute error bound: {}".format(print4F(abError)))
        print("\tRelative error bound: {}".format(print4F(reError)))
        print("\tNumber of paths: ", nr_of_paths)

    if args.operation == "estimationdistribution":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        states = M.states()
        N = int(nrOfSteps(args.numberOfSteps))
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
        distribution, intervals, abError, reError, nr_of_paths, stop = M.estimationDistribution(C, N)
        print("Simulation termination reason: {}".format(stop))
        print("The estimated distribution after {} steps of [{}] is as follows:".format(N, ", ".join(states)))
        print("\tDistribution: {}".format(printList(distribution)))
        print("\tConfidence intervals: {}".format(printDList(intervals)))
        print("\tAbsolute error bound: {}".format(print4F(abError)))
        print("\tRelative error bound: {}".format(print4F(reError)))
        print("\tNumber of paths: ", nr_of_paths)

    if args.operation == "estimationhittingstate":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        if args.stateStartingSet is not None:
            S = [s.strip() for s in args.stateStartingSet.split(',')]
        else:
            S = M.states()
        s = args.targetState
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
        hitting_probability,nr_of_paths,abErrors,reErrors,intervals,stop = M.estimationHittingState(C, s, True, True, S)
        print("Estimated hitting probabilities for {} are:".format(s))
        for i in range(len(hitting_probability)):
            print("f({}, {}) = {}\tint:{}\tabEr:{}\treEr:{}\t#paths:{}\tstop:{}".format(
                S[i], s, print4F(hitting_probability[i]), printList(intervals[i]), 
                print4F(abErrors[i]), print4F(reErrors[i]), nr_of_paths[i], stop[i]
            ))
                
    if args.operation == "estimationhittingreward":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        if args.stateStartingSet is not None:
            S = [s.strip() for s in args.stateStartingSet.split(',')]
        else:
            S = M.states()
        s = args.targetState
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
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
    
    if args.operation == "estimationhittingstateset":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        if args.stateStartingSet is not None:
            S = [s.strip() for s in args.stateStartingSet.split(',')]
        else:
            S = M.states()
        s = [s.strip() for s in args.targetStateSet.split(',')]
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
        hitting_probability,nr_of_paths,abErrors,reErrors,intervals,stop = M.estimationHittingState(C, s, False, True, S)
        print("Estimated hitting probabilities for {{{}}} are:".format(', '.join(s)))
        for i in range(len(hitting_probability)):
            print("f({}, {{{}}}) = {}\tint:{}\tabEr:{}\treEr:{}\t#paths:{}\tstop:{}".format(
                S[i], ', '.join(s), print4F(hitting_probability[i]),
                printList(intervals[i]), print4F(abErrors[i]), print4F(reErrors[i]), nr_of_paths[i], stop[i]
            ))

    if args.operation == "estimationhittingrewardset":
        if args.Seed is not None:
            M.setSeed(int(args.Seed))
        if args.stateStartingSet is not None:
            S = [s.strip() for s in args.stateStartingSet.split(',')]
        else:
            S = M.states()
        s = [s.strip() for s in args.targetStateSet.split(',')]
        C = stopCriteria([stringToFloat(i, -1.0) for i in args.Conditions[1:-1].split(',')])
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