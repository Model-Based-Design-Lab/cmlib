# operations on Markov Chains

OP_DTMC_LIST_STATES = "liststates"
OP_DTMC_LIST_RECURRENT_STATES = "listrecurrentstates"
OP_DTMC_LIST_TRANSIENT_STATES = "listtransientstates"
OP_DTMC_COMMUNICATINGSTATES = "communicatingstates"
OP_DTMC_CLASSIFY_TRANSIENT_RECURRENT = "classifytransientrecurrent"
OP_DTMC_HITTING_PROBABILITY = "hittingprobability"
OP_DTMC_HITTING_PROBABILITY_SET = "hittingprobabilityset"
OP_DTMC_REWARD_TILL_HIT = "rewardtillhit"
OP_DTMC_REWARD_TILL_HIT_SET = "rewardtillhitset"
OP_DTMC_PERIODICITY = "periodicity"
OP_DTMC_MC_TYPE = "mctype"
OP_DTMC_TRANSIENT = "transient"
OP_DTMC_TRANSIENT_REWARDS = "transientRewards"
OP_DTMC_TRANSIENT_MATRIX = "transientMatrix"
OP_DTMC_LIMITING_MATRIX = "limitingMatrix"
OP_DTMC_LIMITING_DISTRIBUTION = "limitingDistribution"
OP_DTMC_LONG_RUN_REWARD = "longRunReward"
OP_DTMC_EXECUTION_GRAPH = "executiongraph"
OP_DTMC_MARKOV_TRACE = "markovtrace"
OP_DTMC_LONG_RUN_EXPECTED_AVERAGE_REWARD = "longrunexpectedaveragereward"
OP_DTMC_CEZARO_LIMIT_DISTRIBUTION = "cezarolimitdistribution"
OP_DTMC_ESTIMATION_EXPECTED_REWARD = "estimationexpectedreward"
OP_DTMC_ESTIMATION_DISTRIBUTION = "estimationdistribution"
OP_DTMC_ESTIMATION_HITTING_STATE = "estimationhittingstate"
OP_DTMC_ESTIMATION_HITTING_REWARD = "estimationhittingreward"
OP_DTMC_ESTIMATION_HITTING_STATE_SET = "estimationhittingstateset"
OP_DTMC_ESTIMATION_HITTING_REWARD_SET = "estimationhittingrewardset"

MarkovChainOperations = [
OP_DTMC_LIST_STATES, OP_DTMC_LIST_RECURRENT_STATES, OP_DTMC_LIST_TRANSIENT_STATES, OP_DTMC_COMMUNICATINGSTATES, OP_DTMC_CLASSIFY_TRANSIENT_RECURRENT, OP_DTMC_HITTING_PROBABILITY, OP_DTMC_HITTING_PROBABILITY_SET, OP_DTMC_REWARD_TILL_HIT, OP_DTMC_REWARD_TILL_HIT_SET, OP_DTMC_PERIODICITY, OP_DTMC_MC_TYPE, OP_DTMC_TRANSIENT, OP_DTMC_TRANSIENT_REWARDS, OP_DTMC_TRANSIENT_MATRIX, OP_DTMC_LIMITING_MATRIX, OP_DTMC_LIMITING_DISTRIBUTION, OP_DTMC_LONG_RUN_REWARD, OP_DTMC_EXECUTION_GRAPH, OP_DTMC_MARKOV_TRACE, OP_DTMC_LONG_RUN_EXPECTED_AVERAGE_REWARD, OP_DTMC_CEZARO_LIMIT_DISTRIBUTION, OP_DTMC_ESTIMATION_EXPECTED_REWARD, OP_DTMC_ESTIMATION_DISTRIBUTION, OP_DTMC_ESTIMATION_HITTING_STATE, OP_DTMC_ESTIMATION_HITTING_REWARD, OP_DTMC_ESTIMATION_HITTING_STATE_SET, OP_DTMC_ESTIMATION_HITTING_REWARD_SET
]

OperationDescriptions = [
        "List all states of the markov chain\n\tNo flags",
        "List all recurrent states of the markov chain\n\tNo flags",
        "List all transient states of the markov chain\n\tNo flags",
        "Provides list of communicating state sets\n\tNo flags",
        "Lists the transient and recurrent states\n\tNo flags",
        "Provides the hitting probability for a specified state\n\trequired flag:\n\t\t[-s, --state]: Target state",
        "Provides the hitting probability for a specified state set\n\trequired flag:\n\t\t[-ss, --stateset]: Set of target states (comma seperated)",
        "Expected reward until hitting specified state\n\trequired flag:\n\t\t[-s, --state]: Target state",
        "Expected reward until hitting specified state set\n\trequired flag:\n\t\t[-ss, --stateset]: Set of target states (comma seperated)",
        "Lists aperiodic and periodic state states\n\tNo flags",
        "Provides type of markov chain: (non-)ergodic (non-)unichain\n\tNo flags",
        "Transient analysis for specified number of steps\n\trequired flag:\n\t\t[-ns, --numberofsteps]: Number of steps",
        "Transient analysis of reward after specified number of steps\n\trequired flag:\n\t\t[-ns, --numberofsteps]: Number of steps",
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
