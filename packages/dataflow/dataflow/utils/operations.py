''' operations on dataflow graphs '''

OP_SDF_INPUT_LABELS = 'inputlabelssdf'
OP_SDF_STATE_LABELS = 'statelabelssdf'
OP_SDF_THROUGHPUT = 'throughput'
OP_SDF_THROUGHPUT_OUTPUT = 'throughputoutput'
OP_SDF_REP_VECTOR = 'repetitionvector'
OP_SDF_LATENCY = 'latency'
OP_SDF_DEADLOCK = 'deadlock'
OP_SDF_GENERALIZED_LATENCY = 'generalizedlatency'
OP_SDF_STATE_MATRIX = 'statematrix'
OP_SDF_STATE_SPACE_REPRESENTATION = 'statespacematrices'
OP_SDF_STATE_MATRIX_MODEL = 'statematrixmodel'
OP_SDF_STATE_SPACE_MATRICES_MODEL = 'statespacematricesmodel'
OP_SDF_GANTT_CHART = 'ganttchart'
OP_SDF_GANTT_CHART_ZERO_BASED = 'ganttchart-zero-based'
OP_SDF_CONVERT_TO_SINGLE_RATE = 'converttosinglerate'
OP_SDF_CONVERT_TO_SDFX = "converttosdfx"

DataflowOperations = [
    OP_SDF_INPUT_LABELS, OP_SDF_STATE_LABELS, OP_SDF_REP_VECTOR, OP_SDF_DEADLOCK,
    OP_SDF_THROUGHPUT, OP_SDF_THROUGHPUT_OUTPUT, OP_SDF_LATENCY, OP_SDF_GENERALIZED_LATENCY,
    OP_SDF_STATE_SPACE_REPRESENTATION, OP_SDF_STATE_MATRIX, OP_SDF_STATE_SPACE_MATRICES_MODEL,
    OP_SDF_STATE_MATRIX_MODEL, OP_SDF_GANTT_CHART, OP_SDF_GANTT_CHART_ZERO_BASED,
    OP_SDF_CONVERT_TO_SINGLE_RATE, OP_SDF_CONVERT_TO_SDFX
    ]

# operations on max-plus models
OP_MPM_EIGENVALUE = 'eigenvalue'
OP_MPM_EVENT_SEQUENCES = 'eventsequences'
OP_MPM_VECTOR_SEQUENCES = 'vectorsequences'
OP_MPM_INPUT_LABELS = 'inputlabelsmpm'
OP_MPM_MATRICES = 'matrices'
OP_MPM_EIGENVECTORS = 'eigenvectors'
OP_MPM_PRECEDENCEGRAPH = 'precedencegraph'
OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ = 'precedencegraphgraphviz'
OP_MPM_STAR_CLOSURE = 'starclosure'
OP_MPM_MULTIPLY = 'multiply'
OP_MPM_MULTIPLY_TRANSFORM = 'multiplytransform'
OP_MPM_VECTOR_TRACE = 'vectortrace'
OP_MPM_VECTOR_TRACE_TRANSFORM = 'vectortracetransform'
OP_MPM_VECTOR_TRACE_XML = 'vectortracexml'
OP_MPM_CONVOLUTION = 'convolution'
OP_MPM_CONVOLUTION_TRANSFORM = 'convolutiontransform'
OP_MPM_MAXIMUM = 'maxsequences'
OP_MPM_MAXIMUM_TRANSFORM = 'maxsequencestransform'
OP_MPM_DELAY_SEQUENCE = 'delaysequence'
OP_MPM_SCALE_SEQUENCE = 'scalesequence'

MPMatrixOperations = [
    OP_MPM_EIGENVALUE, OP_MPM_EVENT_SEQUENCES, OP_MPM_VECTOR_SEQUENCES,
    OP_MPM_INPUT_LABELS, OP_MPM_MATRICES, OP_MPM_EIGENVECTORS, OP_MPM_PRECEDENCEGRAPH,
    OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ, OP_MPM_STAR_CLOSURE, OP_MPM_MULTIPLY,
    OP_MPM_MULTIPLY_TRANSFORM, OP_MPM_VECTOR_TRACE, OP_MPM_VECTOR_TRACE_TRANSFORM,
    OP_MPM_VECTOR_TRACE_XML, OP_MPM_CONVOLUTION, OP_MPM_CONVOLUTION_TRANSFORM,  OP_MPM_MAXIMUM,
    OP_MPM_MAXIMUM_TRANSFORM, OP_MPM_DELAY_SEQUENCE, OP_MPM_SCALE_SEQUENCE
    ]

# other operations
OtherOperations = []

Operations = DataflowOperations + MPMatrixOperations + OtherOperations

OperationDescriptions = [
    'determines inputs of the graph',
    'determine the labels of the elements of the state vector',
    'computes repetition vector',
    'checks graph for deadlock',
    'computes throughput' ,
    'compute latency, requires period and optional initial state',
    'compute generalized latency, requires period',
    'compute state space representation of the graph',
    'compute the state matrix of the graph',
    'compute state space representation of the graph as a new model',
    'compute the state matrix of the graph as a new model',
    'make a Gantt chart of the graph as an XML file for the cmtrace tool ' \
        '(https://github.com/Model-Based-Design-Lab/cmtrace)',
    'make a Gantt chart of the graph as an XML file for the cmtrace tool ' \
        '(https://github.com/Model-Based-Design-Lab/cmtrace) assuming actor firings cannot ' \
        'start before time 0',
    'convert to a single rate graph',
    'convert to an SDFX graph in the SDF3 XML format',
    'compute the largest eigenvalue of a matrix',
    'list event sequences defined in the model',
    'list vector sequences defined in the model',
    'determine the inputs of the model',
    'list matrices defined in the model',
    'compute the eigenvectors of a matrix',
    'compute the precedence graph of a matrix',
    'compute the precedence graph of a matrix as a Graphviz model',
    'compute the star closure of a matrix',
    'multiply matrices and / or vector sequence; requires matrices possible a vector sequence',
    'multiply matrices and / or vector sequence and make a new model with the result, ' \
        'requires matrices possible a vector sequence',
    'compute a vector trace for a state matrix or a set of state-space matrices; ' \
        'optional numberofiterations, optional initialstate, optional sequences',
    'compute a vector trace for a state matrix or a set of state-space matrices as a new ' \
        'max-plus model; optional numberofiterations, optional initialstate, optional sequences',
    'compute a vector trace for a state matrix or a set of state-space matrices as an XML file ' \
        'for the cmtrace tool (https://github.com/Model-Based-Design-Lab/cmtrace); optional ' \
        'numberofiterations, optional initialstate, optional sequences)',
    'compute the convolution of a series of event sequences',
    'compute the convolution of a series of event sequences as a new model',
    'compute the maximum of a series of event sequences',
    'compute the maximum of a series of event sequences as a new model',
    'delay an event sequence by a number of event samples; requires sequence and parameter',
    'scale an event sequence, i.e., add a constant to all event time stamps; requires ' \
        'sequence and parameter'
]
