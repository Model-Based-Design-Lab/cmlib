# operations on dataflow graphs

OP_SDF_INPUT_LABELS = 'inputlabelssdf'
OP_SDF_STATE_LABELS = 'statelabelssdf'
OP_SDF_THROUGHPUT = 'throughput'
OP_SDF_REP_VECTOR = 'repetitionvector'
OP_SDF_LATENCY = 'latency'
OP_SDF_DEADLOCK = 'deadlock'
OP_SDF_GENERALIZED_LATENCY = 'generalizedlatency'
OP_SDF_STATE_MATRIX = 'statematrix'
OP_SDF_STATE_SPACE_REPRESENTATION = 'statespacerepresentation'
OP_SDF_STATE_SPACE_MATRICES = 'statespacematrices'
OP_SDF_GANTT_CHART = 'ganttchart'
OP_SDF_GANTT_CHART_ZERO_BASED = 'ganttchart-zero-based'
OP_SDF_CONVERT_TO_SINGLE_RATE = 'converttosinglerate'

DataflowOperations = [OP_SDF_INPUT_LABELS, OP_SDF_STATE_LABELS, OP_SDF_REP_VECTOR, OP_SDF_DEADLOCK, OP_SDF_THROUGHPUT, OP_SDF_LATENCY, OP_SDF_GENERALIZED_LATENCY, OP_SDF_STATE_SPACE_REPRESENTATION, OP_SDF_STATE_MATRIX, OP_SDF_STATE_SPACE_MATRICES, OP_SDF_GANTT_CHART, OP_SDF_GANTT_CHART_ZERO_BASED, OP_SDF_CONVERT_TO_SINGLE_RATE]


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

MPMatrixOperations = [OP_MPM_EIGENVALUE, OP_MPM_EVENT_SEQUENCES, OP_MPM_VECTOR_SEQUENCES, OP_MPM_INPUT_LABELS, OP_MPM_MATRICES, OP_MPM_EIGENVECTORS, OP_MPM_PRECEDENCEGRAPH, OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ, OP_MPM_STAR_CLOSURE, OP_MPM_MULTIPLY, OP_MPM_MULTIPLY_TRANSFORM, OP_MPM_VECTOR_TRACE, OP_MPM_VECTOR_TRACE_TRANSFORM, OP_MPM_VECTOR_TRACE_XML, OP_MPM_CONVOLUTION, OP_MPM_CONVOLUTION_TRANSFORM,  OP_MPM_MAXIMUM,  OP_MPM_MAXIMUM_TRANSFORM, OP_MPM_DELAY_SEQUENCE, OP_MPM_SCALE_SEQUENCE]

# other operations
OtherOperations = []

Operations = DataflowOperations + MPMatrixOperations + OtherOperations

OperationDescriptions = [
    OP_SDF_INPUT_LABELS + ' (determines inputs of the graph)',
    OP_SDF_STATE_LABELS + ' (determine the labels of the elements of the state vector)',
    OP_SDF_THROUGHPUT + ' (computes throughput)' ,
    OP_SDF_LATENCY + ' (requires period and optional initial state)',
    OP_SDF_REP_VECTOR + '(computes repetition vector)',
    OP_SDF_DEADLOCK + ' (checks graph for deadlock)',
    OP_SDF_GENERALIZED_LATENCY + ' (compute generalized latency, requires period)',
    OP_SDF_STATE_MATRIX + ' (compute the state matrix of the graph',
    OP_SDF_STATE_SPACE_REPRESENTATION + ' (compute state space representation of the graph)',
    OP_SDF_STATE_SPACE_MATRICES + ' (compute state space representation of the graph as a new model)',
    OP_SDF_GANTT_CHART + ' (make a Gantt chart of the graph as an XML file for the sdf3trace2svg tool)',
    OP_SDF_GANTT_CHART_ZERO_BASED + ' (make a Gantt chart of the graph as an XML file for the sdf3trace2svg tool) assuming actor firings cannot start before time 0',
    OP_SDF_CONVERT_TO_SINGLE_RATE + ' (convert to a single rate graph)', 
    OP_MPM_VECTOR_TRACE + ' (compute a vector trace for a state matrix or a set of state-space matrices; optional numberofiterations, optional initialstate, optional sequences)',
    OP_MPM_VECTOR_TRACE_TRANSFORM + ' (compute a vector trace for a state matrix or a set of state-space matrices as a new max-plus model; optional numberofiterations, optional initialstate, optional sequences)',
    OP_MPM_VECTOR_TRACE_XML + ' (compute a vector trace for a state matrix or a set of state-space matrices as an XML file for the sdf3trace2svg tool; optional numberofiterations, optional initialstate, optional sequences)',
    OP_MPM_MATRICES+ ' (list matrices defined in the model)',
    OP_MPM_VECTOR_SEQUENCES+ ' (list vector sequences defined in the model)',
    OP_MPM_EVENT_SEQUENCES + ' (list event sequences defined in the model)',
    OP_MPM_INPUT_LABELS + ' (determine the inputs of the model)',
    OP_MPM_EIGENVALUE + ' (compute the largest eigenvalue of a matrix)',
    OP_MPM_EIGENVECTORS + ' (compute the eigenvectors of a matrix)',
    OP_MPM_PRECEDENCEGRAPH + ' (compute the precedence graph of a matrix)',
    OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ + ' (compute the precedence graph of a matrix as a Graphviz model)',
    OP_MPM_STAR_CLOSURE + ' (compute the star closure of a matrix)',
    OP_MPM_MULTIPLY + ' (multiply matrices and / or vector sequence; requires matrices possible a vector sequence)',
    OP_MPM_MULTIPLY_TRANSFORM + ' (multiply matrices and / or vector sequence and make a new model with the result, requires matrices possible a vector sequence)',
    OP_MPM_CONVOLUTION + ' (compute the convolution of a series of event sequences)',
    OP_MPM_CONVOLUTION_TRANSFORM + ' (compute the convolution of a series of event sequences as a new model)',
    OP_MPM_MAXIMUM + ' (compute the maximum of a series of event sequences)',
    OP_MPM_MAXIMUM_TRANSFORM + ' (compute the maximum of a series of event sequences as a new model)',
    OP_MPM_DELAY_SEQUENCE + ' (delay an event sequence by a number of event samples; requires sequence and parameter)',
    OP_MPM_SCALE_SEQUENCE + ' (scale an event sequence, i.e., add a costant to all event time stamps; requires sequence and parameter)'
]
