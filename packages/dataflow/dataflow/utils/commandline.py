
'''Operations on dataflow '''

import argparse
from dataflow.libsdf import DataflowGraph
from dataflow.libmpm import MaxPlusMatrixModel, VectorSequenceModel
from dataflow.utils.utils import printXmlTrace, printXmlGanttChart, parseInputTraces, parseInitialState, requireNumberOfIterations, parseNumberOfIterations, requirePeriod, getSquareMatrix, requireSequenceOfMatricesAndPossiblyVectorSequence, determineStateSpaceLabels, parseSequences, validateEventSequences, requireParameterInteger, requireOneEventSequence, requireParameterMPValue
from dataflow.maxplus.maxplus import printMPMatrix, mpVector, mpElement, mpNumberOfColumns, mpMinusInfVector, mpMultiplyMatrixVector, mpTransposeMatrix, mpSplitSequence
from dataflow.utils.operations import DataflowOperations, MPMatrixOperations, Operations, OperationDescriptions, OP_SDF_THROUGHPUT, OP_SDF_DEADLOCK, OP_SDF_REP_VECTOR, OP_SDF_LATENCY, OP_SDF_GENERALIZED_LATENCY, OP_SDF_STATE_SPACE_REPRESENTATION, OP_SDF_STATE_MATRIX, OP_SDF_CONVERT_TO_SINGLE_RATE, OP_SDF_STATE_SPACE_MATRICES, OP_SDF_GANTT_CHART, OP_SDF_GANTT_CHART_ZERO_BASED, OP_MPM_EVENT_SEQUENCES, OP_MPM_VECTOR_SEQUENCES, OP_MPM_MATRICES, OP_MPM_EIGENVALUE, OP_MPM_EIGENVECTORS, OP_MPM_PRECEDENCEGRAPH, OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ, OP_MPM_STAR_CLOSURE, OP_MPM_MULTIPLY, OP_MPM_MULTIPLY_TRANSFORM, OP_MPM_VECTOR_TRACE, OP_MPM_VECTOR_TRACE_TRANSFORM, OP_MPM_VECTOR_TRACE_XML,OP_MPM_CONVOLUTION, OP_MPM_CONVOLUTION_TRANSFORM, OP_MPM_MAXIMUM, OP_MPM_MAXIMUM_TRANSFORM, OP_MPM_DELAY_SEQUENCE, OP_MPM_SCALE_SEQUENCE, OP_MPM_INPUT_LABELS, OP_SDF_INPUT_LABELS, OP_SDF_STATE_LABELS
import sys
import re


def main():

    parser = argparse.ArgumentParser(
        description='Perform operations on dataflow graphs.\nhttps://computationalmodeling.info')
    parser.add_argument('dataflow_graph_or_mpmatrix', help="the dataflow graph or max-plus matrix to analyze")
    parser.add_argument('-op', '--operation', dest='operation',
                        help="the operation or analysis to perform, one of : {}".format("; \n".join(OperationDescriptions)))
    parser.add_argument('-p', '--period', dest='period',
                        help="the period of the system (for latency)")
    parser.add_argument('-is', '--initialstate', dest='initialstate',
                        help="the initial state of the system")
    parser.add_argument('-it', '--inputtrace', dest='inputtrace',
                        help="the input trace to the system, a comma-separated list of: (ID=)?([...]|ID)")
    parser.add_argument('-ma', '--matrices', dest='matrices',
                        help="the matrices to operate on as a comma separated list")
    parser.add_argument('-sq', '--sequences', dest='sequences',
                        help="the sequences to operate on")
    parser.add_argument('-pa', '--parameter', dest='parameter',
                        help="parmeter for the operation")
    parser.add_argument('-ni', '--numberofiterations', dest='numberofiterations',
                        help="number of iterations to analyze")
    parser.add_argument('-og', '--outputgraph', dest='outputGraph',
                        help="the outputfile to write output graph to")

    args = parser.parse_args()

    if args.operation not in Operations:
        sys.stderr.write("Unknown operation: {}\n".format(args.operation))
        sys.stderr.write("Operation should be one of: {}.\n".format(", ".join(Operations)))
        exit(1)

    if args.dataflow_graph_or_mpmatrix:
        try:
            with open(args.dataflow_graph_or_mpmatrix, 'r') as sdfMpmFile:
                dsl = sdfMpmFile.read()
        except FileNotFoundError as e:
            sys.stderr.write("File does not exist: {}.\n".format(args.dataflow_graph_or_mpmatrix))
            exit(1)

    try:
        process(args, dsl)
    except Exception as e:
        sys.stderr.write("An error occurred: {}\n".format(e))
        # in final version comment out following line
        # raise e
        exit(1)

    exit(0)


def process(args, dsl):

    if args.operation not in Operations:
        print("Unknown operation or no operation provided")
        print("Operation should be one of: {}.".format(", \n".join(Operations)))
        exit(1)

    if args.operation in DataflowOperations:
        processDataflowOperation(args, dsl)

    if args.operation in MPMatrixOperations:
        processMaxPlusOperation(args, dsl)


def processDataflowOperation(args, dsl):

    # parse the model
    name, G = DataflowGraph.fromDSL(dsl)
    G.validate()

    # execute the selected operation
    # python has no switch statement :(

    # inputlabels
    if args.operation == OP_SDF_INPUT_LABELS:
        print(",".join(G.inputs()))

    # statelabels
    if args.operation == OP_SDF_STATE_LABELS:
        print(",".join(G.stateElementLabels()))


    # throughput
    if args.operation == OP_SDF_THROUGHPUT:
        print('Throughput:')
        print(G.throughput())

    # repetitionvector
    if args.operation == OP_SDF_REP_VECTOR:
        print('Repetition Vector:')
        rates = G.repetitionVector()
        if rates is None:
            print('The graph is inconsistent.')
        else:
            for a in G.actors():
                print('{}: {}'.format(a, rates[a]))

    # deadlock
    if args.operation == OP_SDF_DEADLOCK:
        if G.deadlock():
            print('The graph deadlocks.')
        else:
            print('The graph does not deadlock.')

    # converttosinglerate
    if args.operation == OP_SDF_CONVERT_TO_SINGLE_RATE:
        GS = G.convertToSingleRate()
        print(GS.asDSL(name+'_singlerate'))

    # latency
    if args.operation == OP_SDF_LATENCY:
        mu = requirePeriod(args)
        x0 = parseInitialState(args, G.numberOfInitialTokens())
        print('Inputs:')
        print(G.listOfInputsStr())
        print('Outputs:')
        print(G.listOfOutputsStr())
        print(G.latency(x0, mu))

    # generalized latency
    if args.operation == OP_SDF_GENERALIZED_LATENCY:
        mu = requirePeriod(args)
        print('Inputs:')
        print(G.listOfInputsStr())
        print('Outputs:')
        print(G.listOfOutputsStr())
        print('State vector:')
        print(G.listOfStateElementsStr())
        LambdaX, LambdaIO = G.generalizedLatency(mu)
        print('IO latency matrix:')
        printMPMatrix(LambdaIO)
        print('Initial state latency matrix:')
        printMPMatrix(LambdaX)

    if args.operation == OP_SDF_STATE_SPACE_REPRESENTATION:
        _, M = G.stateSpaceMatrices()
        print('Inputs:')
        print(G.listOfInputsStr())
        print('Outputs:')
        print(G.listOfOutputsStr())
        print('State vector:')
        print(G.listOfStateElementsStr())
        print()
        print('State matrix A:')
        printMPMatrix(M[0])
        print()
        print('Input matrix B:')
        printMPMatrix(M[1])
        print()
        print('Output matrix C:')
        printMPMatrix(M[2])
        print()
        print('Feed forward matrix D:')
        printMPMatrix(M[3])

    if args.operation == OP_SDF_STATE_MATRIX:
        _, SSM = G.stateSpaceMatrices()
        mpm = MaxPlusMatrixModel()
        mpm.setMatrix(SSM[0])
        matrices = dict()
        matrices['A'] = MaxPlusMatrixModel(SSM[0])
        matrices['A'].setLabels(G.stateElementLabels())
        print(mpm.asDSL(name+"_MPM", matrices))

    if args.operation == OP_SDF_STATE_SPACE_MATRICES:
        _, SSM = G.stateSpaceMatrices()
        mpm = MaxPlusMatrixModel()
        mpm.setMatrix(SSM[0])
        matrices = dict()
        matrices['A'] = MaxPlusMatrixModel(SSM[0])
        matrices['A'].setLabels(G.stateElementLabels())
        matrices['B'] = MaxPlusMatrixModel(SSM[1])
        matrices['B'].setLabels(G.stateElementLabels() + G.inputs())
        matrices['C'] = MaxPlusMatrixModel(SSM[2])
        matrices['C'].setLabels(G.outputs() + G.stateElementLabels())
        matrices['D'] = MaxPlusMatrixModel(SSM[3])
        matrices['D'].setLabels(G.outputs() + G.inputs())
        print(mpm.asDSL(name+"_MPM", matrices))

    if args.operation == OP_SDF_GANTT_CHART:
        ni = requireNumberOfIterations(args)
        inputTraces, outputTraces, firingStarts, firingDurations = _determineTrace(G, args, ni)

        # write gantt chart trace
        printXmlGanttChart(G.actorsWithoutInputsOutputs(), G.repetitionVector(), firingStarts, firingDurations, G.inputs(), inputTraces, G.outputs(), outputTraces)

    if args.operation == OP_SDF_GANTT_CHART_ZERO_BASED:
        # make a Gantt chart assuming that actors cannot fire before time 0
        # use artificial inputs to all actors and remove them later

        realInputs = list(G.inputs())
        ni = requireNumberOfIterations(args)

        # create name for artificial input to actor a
        inpName = lambda a: '_zb_{}'.format(a)

        # determine the repetition vector for the extended graph
        reps = G.repetitionVector()

        # add the new inputs and channels
        for a in G.actorsWithoutInputsOutputs():
            G.addInputPort(inpName(a))
            G.addChannel(inpName(a), a, dict())
            # provide the new inputs with input event sequences of sufficient zeros
            # add signal of number of iterations times the repetition vector of the actor consuming from the input
            G.addInputSignal(inpName(a), [0] * ni * reps[a])

        inputTraces, outputTraces, firingStarts, firingDurations = _determineTrace(G, args, ni)

        # suppress the artificial inputs
        num = len(G.actorsWithoutInputsOutputs())
        reduceRealInputs = lambda l: l[:-num]
        realInputTraces = list(map(reduceRealInputs, inputTraces))

        # write gantt chart trace
        printXmlGanttChart(G.actorsWithoutInputsOutputs(), reps, firingStarts, firingDurations, realInputs, realInputTraces, G.outputs(), outputTraces)


def processMaxPlusOperation(args, dsl):

    name, Matrices, VectorSequences, EventSequences  = MaxPlusMatrixModel.fromDSL(dsl)
    for m in Matrices.values():
        m.validate()
    for v in VectorSequences.values():
        v.validate()
    for e in EventSequences.values():
        e.validate()

    # eventsequences
    if args.operation == OP_MPM_EVENT_SEQUENCES:
        print(",".join(EventSequences.keys()))
    
    # vectorsequences
    if args.operation == OP_MPM_VECTOR_SEQUENCES:
        print(",".join(VectorSequences.keys()))

    # matrices
    if args.operation == OP_MPM_MATRICES:
        print(",".join(Matrices.keys()))

    # eigenvalue
    if args.operation == OP_MPM_EIGENVALUE:
        mat = getSquareMatrix(Matrices, args)
        print("The largest eigenvalue of matrix {} is:".format(mat))
        print(Matrices[mat].eigenvalue())

    # eigenvectors
    if args.operation == OP_MPM_EIGENVECTORS:
        mat = getSquareMatrix(Matrices, args)
        (ev, gev) = Matrices[mat].eigenvectors()
        print("The eigenvectors of matrix {} are:".format(mat))
        if len(ev)==0:
            print('None')
        else:
            for v in ev:
                print('{}, with eigenvalue: {:.2f}'.format(mpVector(v[0]), v[1]))
        if len(gev) > 0:
            print('\nGeneralized Eigenvectors:')
            for v in gev:
                print('{}, with generalized eigenvalue: {}'.format(mpVector(v[0]), mpVector(v[1])))

    # precedence graph
    if args.operation == OP_MPM_PRECEDENCEGRAPH:
        mat = getSquareMatrix(Matrices, args)
        g = Matrices[mat].precedencegraph()
        print("The nodes of the precedence graph are:")
        print(", ".join(g.nodes()))
        print("The edges of the precedence graph are:")
        for e in g.edges():
            print("{} --- {} ---> {}".format(e[0], g.edge_weight(e), e[1]))

    # precedence graph graphviz
    if args.operation == OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ:
        mat = getSquareMatrix(Matrices, args)
        g = Matrices[mat].precedencegraphGraphviz()
        print(g)

    # star closure
    if args.operation == OP_MPM_STAR_CLOSURE:
        mat = getSquareMatrix(Matrices, args)
        success, cl = Matrices[mat].starClosure()
        if success:
            printMPMatrix(cl)
        else:
            print("The matrix has no star closure.")

    # multiply 
    if args.operation == OP_MPM_MULTIPLY:
        names = requireSequenceOfMatricesAndPossiblyVectorSequence(Matrices, VectorSequences, args)
        matrices = [(Matrices[m] if m in Matrices else VectorSequences[m]) for m in names]
        result = MaxPlusMatrixModel.multiplySequence(matrices)
        print("The product of {} is:".format(', '.join(names)))
        printMPMatrix(result.mpMatrix())

    # multiplytransform 
    if args.operation == OP_MPM_MULTIPLY_TRANSFORM:
        names = requireSequenceOfMatricesAndPossiblyVectorSequence(Matrices, VectorSequences, args)
        matrices = [(Matrices[m] if m in Matrices else VectorSequences[m]) for m in names]
        newName = 'prod_{}'.format('_'.join(names))
        result = MaxPlusMatrixModel.multiplySequence(matrices)
        newModel = dict()
        newModel[newName] = result
        print(MaxPlusMatrixModel().asDSL(name+'_mul', newModel))

    # inputlabels
    if args.operation == OP_MPM_INPUT_LABELS:
        inputLabels = _determineInputLabels(Matrices)
        print(",".join(inputLabels))


    # vectortrace
    # - on a model with a single, square matrix, compute a sequence of state vectors for the given 
    # number of iterations, including the initial state
    # - on a model with A, B, C and D matrices, determine inputs from 
    #  * the command-line inputtrace spec
    #  * event sequences or vector sequences defined in the model
    if args.operation == OP_MPM_VECTOR_TRACE:
        labels, vt = _makeVectorTrace(Matrices, VectorSequences, EventSequences, args)
        print('Vector elements: [{}]'.format(', '.join(labels)))
        print('Trace:')
        print(', '.join([mpVector(v) for v in vt]))
    
    # vectortracetransform
    if args.operation == OP_MPM_VECTOR_TRACE_TRANSFORM:
        labels, vt = _makeVectorTrace(Matrices, VectorSequences, EventSequences, args)        
        res = dict()
        vsm = VectorSequenceModel()
        for v in vt:
            vsm.addVector(v)
        vsm.setLabels(labels)
        res[name] = vsm
        print(MaxPlusMatrixModel().asDSL(name+'_trace', res))

    # vectortracexml
    if args.operation == OP_MPM_VECTOR_TRACE_XML:

        ni = parseNumberOfIterations(args)
        sequences = parseSequences(args)

        if len(sequences) ==0:
            # nothing was specified on the command line, use all vector sequences and event sequences as default
            for s in VectorSequences:
                sequences.append(s)
            for s in EventSequences:
                sequences.append(s)
            if len(sequences) == 0:
                # still nothing?
                raise Exception("vectortracexml requires sequences.")

        # determine the labels and the final legnth of the trace
        tracelen = ni
        labels = []
        for s in sequences:
            if not (s in VectorSequences or s in EventSequences):
                raise Exception("Unknown vector or event sequence {}.".format(s))
            if s in VectorSequences:
                vs = VectorSequences[s]
                for n in range(vs.vectorLength()):
                    labels.append(vs.getLabel(n, s))
                tracelen = vs.length() if tracelen is None else min(tracelen, vs.length())
            elif s in EventSequences:
                ms = EventSequences[s]
                labels.append(s)
                tracelen = ms.length() if tracelen is None else min(tracelen, ms.length())
            else:
                raise Exception("Sequence {} is unknown.".format(s))

        # collect the actual trace
        vt = []
        for s in sequences:
            if s in VectorSequences:
                vs = VectorSequences[s]
                for r in mpTransposeMatrix(vs.vectors()):
                    vt.append(r[:tracelen])
            else:
                es = EventSequences[s]
                vt.append(es.sequence()[:tracelen])

        # transpose the result
        vt = mpTransposeMatrix(vt)

        printXmlTrace(vt, labels)

    # convolution
    if args.operation == OP_MPM_CONVOLUTION:
        sequences, res = _convolution(EventSequences, args)
        print("The convolution of {} is:".format(", ".join(sequences)))
        print(res)

    # convolutiontransform
    if args.operation == OP_MPM_CONVOLUTION_TRANSFORM:
        sequences, ress = _convolution(EventSequences, args)
        res = dict()
        res['{}_conv'.format('_'.join(sequences))] = ress
        print(MaxPlusMatrixModel().asDSL(name+'_conv', res))

    # maxsequences
    if args.operation == OP_MPM_MAXIMUM:
        sequences, res = _maximum(EventSequences, args)
        print("The maximum of {} is:".format(", ".join(sequences)))
        print(res)

    # maxsequencestransform
    if args.operation == OP_MPM_MAXIMUM_TRANSFORM:
        sequences, ress = _maximum(EventSequences, args)
        res = dict()
        res['{}_max'.format('_'.join(sequences))] = ress
        print(MaxPlusMatrixModel().asDSL(name+'_max', res))

    #'delaysequence'
    if args.operation == OP_MPM_DELAY_SEQUENCE:
        delay = requireParameterInteger(args)
        seq = requireOneEventSequence(EventSequences, args)
        res = EventSequences[seq].delay(delay)
        print("The {}-delayed sequence of {} is:".format(delay, seq))
        print(res)

    #'scalesequence'
    if args.operation == OP_MPM_SCALE_SEQUENCE:
        scale = requireParameterMPValue(args)
        seq = requireOneEventSequence(EventSequences, args)
        res = EventSequences[seq].scale(scale)
        print("The scaled sequence of {} by scaling factor {} is:".format(seq, mpElement(scale)))
        print(res)



def _determineInputLabels(matrices):
    if len(matrices) == 1:
        return []
    else:
        inputLabels, _, _ = determineStateSpaceLabels(matrices)
        return inputLabels


def _makeVectorTrace(matrices, vectorsequences, eventsequences, args):
    if len(matrices) == 1:
        ni = requireNumberOfIterations(args)
        M =matrices.values()[0] 
        if not M.isSquare():
            raise Exception("Matrix must be square.")
        x0 = parseInitialState(args, M.numberOfRows())
        vt = M.vectortraceClosed(x0, ni)
        inputs = []
        stateSize = M.numberOfRows()
        inputLabels = []
        outputLabels = []
        stateLabels = []
    else:
        ni = parseNumberOfIterations(args)
        inputLabels, stateLabels, outputLabels = determineStateSpaceLabels(matrices)            
        stateSize = len(stateLabels)
        x0 = parseInitialState(args, stateSize)
        nt, ut = parseInputTraces(eventsequences, vectorsequences, args)
        inputs = MaxPlusMatrixModel.extractSequences(nt, ut, eventsequences, vectorsequences, inputLabels)
        vt = MaxPlusMatrixModel.vectortrace(matrices, x0, ni, inputs, True)

    labels = []
    labels = labels + inputLabels
    labels = labels + stateLabels
    labels = labels + outputLabels
    
    return labels, vt

def _convolution(eventsequences, args):
    sequences = parseSequences(args)
    if len(sequences) < 2:
        raise Exception("Please specify at least two sequences to convolve.")
    validateEventSequences(eventsequences, sequences)

    res = eventsequences[sequences[0]]
    for s in sequences[1:]:
        res = res.convolveWith(eventsequences[s])
    return sequences, res

def _maximum(eventsequences, args):
    sequences = parseSequences(args)
    if len(sequences) < 2:
        raise Exception("Please specify at least two sequences to maximize.")
    validateEventSequences(eventsequences, sequences)

    res = eventsequences[sequences[0]]
    for s in sequences[1:]:
        res = res.maxWith(eventsequences[s])
    return sequences, res

def _determineTrace(G: DataflowGraph, args, ni):
    stateSize = G.numberOfInitialTokens()
    x0 = parseInitialState(args, stateSize)

    # get input sequences.
    nt, _ = parseInputTraces(G.inputSignals(), {}, args)

    if stateSize != len(x0):
        raise Exception('Initial state vector is of incorrect size.')

    return G.determineTrace(ni, x0, nt)


if __name__ == "__main__":
    main()
