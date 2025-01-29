
'''Operations on dataflow '''

import argparse
import sys
from fractions import Fraction
from typing import Any, Dict

from dataflow.libmpm import MaxPlusMatrixModel, VectorSequenceModel
from dataflow.libsdf import DataflowGraph
from dataflow.maxplus.maxplus import mp_transpose_matrix
from dataflow.maxplus.utils.printing import (mp_element_to_string, mp_pretty_value,
                                             mp_pretty_vector_to_string,
                                             pretty_print_mp_matrix,
                                             print_mp_vector_list)
from dataflow.utils.operations import (OP_MPM_CONVOLUTION,
                                       OP_MPM_CONVOLUTION_TRANSFORM,
                                       OP_MPM_DELAY_SEQUENCE,
                                       OP_MPM_EIGENVALUE, OP_MPM_EIGENVECTORS,
                                       OP_MPM_EVENT_SEQUENCES,
                                       OP_MPM_INPUT_LABELS, OP_MPM_MATRICES,
                                       OP_MPM_MAXIMUM,
                                       OP_MPM_MAXIMUM_TRANSFORM,
                                       OP_MPM_MULTIPLY,
                                       OP_MPM_MULTIPLY_TRANSFORM,
                                       OP_MPM_PRECEDENCEGRAPH,
                                       OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ,
                                       OP_MPM_SCALE_SEQUENCE,
                                       OP_MPM_STAR_CLOSURE,
                                       OP_MPM_VECTOR_SEQUENCES,
                                       OP_MPM_VECTOR_TRACE,
                                       OP_MPM_VECTOR_TRACE_TRANSFORM,
                                       OP_MPM_VECTOR_TRACE_XML,
                                       OP_SDF_CONVERT_TO_SINGLE_RATE,
                                        OP_SDF_CONVERT_TO_SDFX,
                                       OP_SDF_DEADLOCK, OP_SDF_GANTT_CHART,
                                       OP_SDF_GANTT_CHART_ZERO_BASED,
                                       OP_SDF_GENERALIZED_LATENCY,
                                       OP_SDF_INPUT_LABELS, OP_SDF_LATENCY,
                                       OP_SDF_REP_VECTOR, OP_SDF_STATE_LABELS,
                                       OP_SDF_STATE_MATRIX,
                                       OP_SDF_STATE_MATRIX_MODEL,
                                       OP_SDF_STATE_SPACE_MATRICES_MODEL,
                                       OP_SDF_STATE_SPACE_REPRESENTATION,
                                       OP_SDF_THROUGHPUT,
                                       OP_SDF_THROUGHPUT_OUTPUT,
                                       DataflowOperations, MPMatrixOperations,
                                       OperationDescriptions, Operations)
from dataflow.utils.utils import (
    DataflowException, determine_state_space_labels, fraction_to_float_list,
    fraction_to_float_optional_l_list, get_square_matrix, parse_initial_state,
    parse_input_traces, parse_number_of_iterations, parse_sequences,
    print_xml_gantt_chart, print_xml_trace, require_number_of_iterations,
    require_one_event_sequence, require_parameter_integer, require_parameter_mp_value,
    require_period, require_sequence_of_matrices_and_possibly_vector_sequence,
    validate_event_sequences)

def main():
    """Main entry point."""

    # optional help flag explaining usage of each individual operation
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('-oh', '--operationhelp', dest='opHelp', nargs="?", const=" ")
    options, remainder = parser.parse_known_args() # Only use options of parser above


    if options.opHelp: # Check if -oh has been called
        if options.opHelp not in Operations:
            if options.opHelp.strip() != '':
                print("Operation '{}' does not exist. List of operations:\n\t- {}".format( \
                    options.opHelp, "\n\t- ".join(Operations)))
            else:
                print("List of operations:\n\t- {}".format("\n\t- ".join(Operations)))
        else:
            print(f"{options.opHelp}: {OperationDescriptions[Operations.index(options.opHelp)]}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Perform operations on dataflow graphs.\n'
                                     'https://computationalmodeling.info')
    parser.add_argument('dataflow_graph_or_mpmatrix', \
                        help="the dataflow graph or max-plus matrix to analyze")
    parser.add_argument('-op', '--operation', dest='operation',
                        help="the operation or analysis to perform, one of : {}.\n" \
                        "Use 'dataflow -oh OPERATION' for information about the " \
                        "specific operation.".format("; \n".join(Operations)))
    parser.add_argument('-p', '--period', dest='period',
                        help="the period of the system (for latency)")
    parser.add_argument('-is', '--initialstate', dest='initialstate',
                        help="the initial state of the system")
    parser.add_argument('-it', '--inputtrace', dest='inputtrace',
                        help="the input trace to the system, a comma-separated " \
                            "list of: (ID=)?([...]|ID)")
    parser.add_argument('-ma', '--matrices', dest='matrices',
                        help="the matrices to operate on as a comma separated list")
    parser.add_argument('-sq', '--sequences', dest='sequences',
                        help="the sequences to operate on")
    parser.add_argument('-pa', '--parameter', dest='parameter',
                        help="parameter for the operation")
    parser.add_argument('-ou', '--output', dest='output',
                        help="selected output for the operation")
    parser.add_argument('-ni', '--numberofiterations', dest='numberofiterations',
                        help="number of iterations to analyze")
    parser.add_argument('-og', '--outputgraph', dest='outputGraph',
                        help="the outputfile to write output graph to")

    args = parser.parse_args(remainder)

    if args.operation not in Operations:
        sys.stderr.write(f"Unknown operation: {args.operation}\n")
        sys.stderr.write(f"Operation should be one of: {', '.join(Operations)}.\n")
        sys.exit(1)

    dsl = None
    if args.dataflow_graph_or_mpmatrix:
        try:
            with open(args.dataflow_graph_or_mpmatrix, 'r', encoding='utf-8') as sdf_mpm_file:
                dsl = sdf_mpm_file.read()
        except FileNotFoundError:
            sys.stderr.write(f"File does not exist: {args.dataflow_graph_or_mpmatrix}.\n")
            sys.exit(1)

    try:
        process(args, dsl)
    except Exception as e: # pylint: disable=broad-exception-caught
        sys.stderr.write(f"{e}\n")
        # in final version comment out following line
        # raise e
        sys.exit(1)

    sys.exit(0)


def process(args, dsl):
    """Process arguments and execute operations."""

    if args.operation not in Operations:
        print("Unknown operation or no operation provided")
        print("Operation should be one of: {}.".format(", \n".join(Operations)))
        sys.exit(1)

    if args.operation in DataflowOperations:
        process_dataflow_operation(args, dsl)

    if args.operation in MPMatrixOperations:
        process_max_plus_operation(args, dsl)


def process_dataflow_operation(args, dsl):
    """Process arguments and execute dataflow operations."""

    # parse the model
    name, dataflow_graph = DataflowGraph.from_dsl(dsl)
    dataflow_graph.validate()

    # execute the selected operation
    # python has no switch statement :(

    # inputlabels
    if args.operation == OP_SDF_INPUT_LABELS:
        print(",".join(dataflow_graph.inputs()))

    # statelabels
    if args.operation == OP_SDF_STATE_LABELS:
        print(",".join(dataflow_graph.state_element_labels()))


    # throughput
    if args.operation == OP_SDF_THROUGHPUT:
        print('Throughput:')
        print(dataflow_graph.throughput())

    # throughput of an output
    if args.operation == OP_SDF_THROUGHPUT_OUTPUT:
        if args.output is None:
            raise DataflowException("Please specify output with -ou option.")
        print(f'Throughput of output: {args.output}:')
        print(dataflow_graph.throughput_output(args.output))


    # repetitionvector
    if args.operation == OP_SDF_REP_VECTOR:
        print('Repetition Vector:')
        rates = dataflow_graph.repetition_vector()
        if isinstance(rates, list):
            print('The graph is inconsistent.')
            print('There is an inconsistent cycle between the following ' \
                   f'actors: {", ".join(rates)}')
        else:
            for a in dataflow_graph.actors():
                print(f'{a}: {rates[a]}')

    # deadlock
    if args.operation == OP_SDF_DEADLOCK:
        if dataflow_graph.deadlock():
            print('The graph deadlocks.')
        else:
            print('The graph does not deadlock.')

    # converttosinglerate
    if args.operation == OP_SDF_CONVERT_TO_SINGLE_RATE:
        dataflow_graph_sr = dataflow_graph.convert_to_single_rate()
        print(dataflow_graph_sr.as_dsl(name+'_singlerate'))

    # converttosdf3 xml
    if args.operation == OP_SDF_CONVERT_TO_SDFX:
        print(dataflow_graph.as_sdfx(name+'_sdfx'))

    # latency
    if args.operation == OP_SDF_LATENCY:
        mu = require_period(args)
        x0 = parse_initial_state(args, dataflow_graph.number_of_initial_tokens())
        print('Inputs:')
        print(dataflow_graph.list_of_inputs_str())
        print('Outputs:')
        print(dataflow_graph.list_of_outputs_str())
        pretty_print_mp_matrix(dataflow_graph.latency(x0, mu))

    # generalized latency
    if args.operation == OP_SDF_GENERALIZED_LATENCY:
        mu = require_period(args)
        print('Inputs:')
        print(dataflow_graph.list_of_inputs_str())
        ivs = len(dataflow_graph.inputs())
        print('Outputs:')
        print(dataflow_graph.list_of_outputs_str())
        ovs = len(dataflow_graph.outputs())
        print('State vector:')
        print(dataflow_graph.list_of_state_elements_str())
        svs = len(dataflow_graph.state_element_labels())
        lambda_x, lambda_io = dataflow_graph.generalized_latency(mu)
        print('IO latency matrix:')
        pretty_print_mp_matrix(lambda_io, ovs, ivs)
        print('Initial state latency matrix:')
        pretty_print_mp_matrix(lambda_x, ovs, svs)

    if args.operation == OP_SDF_STATE_MATRIX:
        _, st_sp_matrices = dataflow_graph.state_space_matrices()
        svl = dataflow_graph.list_of_state_elements_str()
        print('State vector:')
        print(svl)
        print()
        print('State matrix A:')
        pretty_print_mp_matrix(st_sp_matrices[0])
        print()

    if args.operation == OP_SDF_STATE_SPACE_REPRESENTATION:
        _, st_sp_matrices = dataflow_graph.state_space_matrices()
        svl = dataflow_graph.list_of_state_elements_str()
        svs = len(dataflow_graph.state_element_labels())
        ivl = dataflow_graph.list_of_inputs_str()
        ivs = len(dataflow_graph.inputs())
        ovl = dataflow_graph.list_of_outputs_str()
        ovs = len(dataflow_graph.outputs())
        print('Inputs:')
        print(ivl)
        print('Outputs:')
        print(ovl)
        print('State vector:')
        print(svl)
        print()
        print('State matrix A:')
        pretty_print_mp_matrix(st_sp_matrices[0])
        print()
        print('Input matrix B:')
        pretty_print_mp_matrix(st_sp_matrices[1], svs, ivs)
        print()
        print('Output matrix C:')
        pretty_print_mp_matrix(st_sp_matrices[2], ovs, svs)
        print()
        print('Feed forward matrix D:')
        pretty_print_mp_matrix(st_sp_matrices[3], ovs, ivs)


    if args.operation == OP_SDF_STATE_MATRIX_MODEL:
        _, st_sp_matrices = dataflow_graph.state_space_matrices()
        mpm = MaxPlusMatrixModel()
        mpm.set_matrix(st_sp_matrices[0])
        matrices = {}
        matrices['A'] = MaxPlusMatrixModel(st_sp_matrices[0])
        matrices['A'].set_labels(dataflow_graph.state_element_labels())
        print(mpm.as_dsl(name+"_MPM", matrices))

    if args.operation == OP_SDF_STATE_SPACE_MATRICES_MODEL:
        _, st_sp_matrices = dataflow_graph.state_space_matrices()
        mpm = MaxPlusMatrixModel()
        mpm.set_matrix(st_sp_matrices[0])
        matrices = {}
        matrices['A'] = MaxPlusMatrixModel(st_sp_matrices[0])
        matrices['A'].set_labels(dataflow_graph.state_element_labels())
        matrices['B'] = MaxPlusMatrixModel(st_sp_matrices[1])
        matrices['B'].set_labels(dataflow_graph.state_element_labels() + dataflow_graph.inputs())
        matrices['C'] = MaxPlusMatrixModel(st_sp_matrices[2])
        matrices['C'].set_labels(dataflow_graph.outputs() + dataflow_graph.state_element_labels())
        matrices['D'] = MaxPlusMatrixModel(st_sp_matrices[3])
        matrices['D'].set_labels(dataflow_graph.outputs() + dataflow_graph.inputs())
        print(mpm.as_dsl(name+"_MPM", matrices))

    if args.operation == OP_SDF_GANTT_CHART:
        ni = require_number_of_iterations(args)
        input_traces, output_traces, firing_starts, firing_durations = \
            _determine_trace(dataflow_graph, args, ni)

        # write gantt chart trace
        rv = dataflow_graph.repetition_vector()
        if isinstance(rv, list):
            raise DataflowException("The graph is inconsistent.")
        float_firing_durations = [float(d) for d in firing_durations]
        print_xml_gantt_chart(dataflow_graph.actors_without_inputs_outputs(), rv, \
                           fraction_to_float_optional_l_list(firing_starts), float_firing_durations, \
                           dataflow_graph.inputs(), fraction_to_float_optional_l_list(input_traces), \
                           dataflow_graph.outputs(), fraction_to_float_optional_l_list(output_traces))

    if args.operation == OP_SDF_GANTT_CHART_ZERO_BASED:
        # make a Gantt chart assuming that actors cannot fire before time 0
        # use artificial inputs to all actors and remove them later

        real_inputs = list(dataflow_graph.inputs())
        ni = require_number_of_iterations(args)

        # create name for artificial input to actor a
        def inp_name(a):
            return f'_zb_{a}'

        # determine the repetition vector for the extended graph
        reps = dataflow_graph.repetition_vector()
        if isinstance(reps, list):
            raise DataflowException("The graph is inconsistent")

        # add the new inputs and channels
        for a in dataflow_graph.actors_without_inputs_outputs():
            dataflow_graph.add_input_port(inp_name(a))
            dataflow_graph.add_channel(inp_name(a), a, {})
            # provide the new inputs with input event sequences of sufficient zeros
            # add signal of number of iterations times the repetition vector of the
            # actor consuming from the input
            dataflow_graph.add_input_signal(inp_name(a), list([Fraction(0.0)] * ni * reps[a]))

        input_traces, output_traces, firing_starts, firing_durations = \
            _determine_trace(dataflow_graph, args, ni)

        # suppress the artificial inputs
        num = len(dataflow_graph.actors_without_inputs_outputs())

        def reduce_real_inputs(l):
            return l[:-num]
        real_input_traces = list(map(reduce_real_inputs, input_traces))

        # write gantt chart trace
        float_firing_durations = [float(d) for d in firing_durations]
        print_xml_gantt_chart(dataflow_graph.actors_without_inputs_outputs(), reps, \
                           fraction_to_float_optional_l_list(firing_starts), fraction_to_float_list \
                           (firing_durations), real_inputs, fraction_to_float_optional_l_list \
                           (real_input_traces), dataflow_graph.outputs(), \
                           fraction_to_float_optional_l_list(output_traces))


def process_max_plus_operation(args, dsl):
    """Process args for maxplus operation."""

    name, matrices, vector_sequences, event_sequences  = MaxPlusMatrixModel.from_dsl(dsl)
    for m in matrices.values():
        m.validate()
    for v in vector_sequences.values():
        v.validate()
    for e in event_sequences.values():
        e.validate()

    # eventsequences
    if args.operation == OP_MPM_EVENT_SEQUENCES:
        print(",".join(event_sequences.keys()))

    # vectorsequences
    if args.operation == OP_MPM_VECTOR_SEQUENCES:
        print(",".join(vector_sequences.keys()))

    # matrices
    if args.operation == OP_MPM_MATRICES:
        print(",".join(matrices.keys()))

    # eigenvalue
    if args.operation == OP_MPM_EIGENVALUE:
        mat = get_square_matrix(matrices, args)
        print(f"The largest eigenvalue of matrix {mat} is:")
        print(mp_pretty_value(matrices[mat].eigenvalue()))

    # eigenvectors
    if args.operation == OP_MPM_EIGENVECTORS:
        mat = get_square_matrix(matrices, args)
        (ev, gev) = matrices[mat].eigenvectors()
        print(f"The eigenvectors of matrix {mat} are:")
        if len(ev)==0:
            print('None')
        else:
            for v in ev:
                print(f'{mp_pretty_vector_to_string(v[0])}, with eigenvalue: {mp_pretty_value(v[1])}')
        if len(gev) > 0:
            print('\nGeneralized Eigenvectors:')
            for v in gev:
                print(f'{mp_pretty_vector_to_string(v[0])}, with generalized eigenvalue: ' \
                      f'{mp_pretty_vector_to_string(v[1])}')

    # precedence graph
    if args.operation == OP_MPM_PRECEDENCEGRAPH:
        mat = get_square_matrix(matrices, args)
        g = matrices[mat].precedence_graph()
        print("The nodes of the precedence graph are:")
        print(", ".join(g.nodes()))
        print("The edges of the precedence graph are:")
        for e in g.edges():
            print(f"{e[0]} --- {g.edge_weight(e)} ---> {e[1]}")

    # precedence graph graphviz
    if args.operation == OP_MPM_PRECEDENCEGRAPH_GRAPHVIZ:
        mat = get_square_matrix(matrices, args)
        g = matrices[mat].precedence_graph_graphviz()
        print(g)

    # star closure
    if args.operation == OP_MPM_STAR_CLOSURE:
        mat = get_square_matrix(matrices, args)
        success, cl = matrices[mat].star_closure()
        if success:
            m: MaxPlusMatrixModel = cl  # type: ignore
            pretty_print_mp_matrix(m.mp_matrix())
        else:
            print("The matrix has no star closure.")

    # multiply
    if args.operation == OP_MPM_MULTIPLY:
        names = require_sequence_of_matrices_and_possibly_vector_sequence(matrices, vector_sequences, args)
        matrices = [(matrices[m] if m in matrices else vector_sequences[m]) for m in names]
        result = MaxPlusMatrixModel.multiply_sequence(matrices)
        print(f"The product of {', '.join(names)} is:")
        if isinstance(result, VectorSequenceModel):
            print_mp_vector_list(result.vectors())
        else:
            pretty_print_mp_matrix(result.mp_matrix())

    # multiplytransform
    if args.operation == OP_MPM_MULTIPLY_TRANSFORM:
        names = require_sequence_of_matrices_and_possibly_vector_sequence(matrices, # type: ignore \
                                                                   vector_sequences, args)
        matrices = [(matrices[m] if m in matrices else vector_sequences[m]) # type: ignore \
                    for m in names]
        new_name = f"prod_{'_'.join(names)}"
        result = MaxPlusMatrixModel.multiply_sequence(matrices)
        new_model = {}
        new_model[new_name] = result
        print(MaxPlusMatrixModel().as_dsl(name+'_mul', new_model))

    # inputlabels
    if args.operation == OP_MPM_INPUT_LABELS:
        input_labels = _determine_input_labels(matrices)
        print(",".join(input_labels))


    # vectortrace
    # - on a model with a single, square matrix, compute a sequence of state vectors for the given
    # number of iterations, including the initial state
    # - on a model with A, B, C and D matrices, determine inputs from
    #  * the command-line inputtrace spec
    #  * event sequences or vector sequences defined in the model
    if args.operation == OP_MPM_VECTOR_TRACE:
        labels, vt = _make_vector_trace(matrices, vector_sequences, event_sequences, args)
        print(f"Vector elements: [{', '.join(labels)}]")
        print('Trace:')
        print(', '.join([mp_pretty_vector_to_string(v) for v in vt]))

    # vectortracetransform
    if args.operation == OP_MPM_VECTOR_TRACE_TRANSFORM:
        labels, vt = _make_vector_trace(matrices, vector_sequences, event_sequences, args)
        res = {}
        vsm = VectorSequenceModel()
        for v in vt:
            vsm.add_vector(v)
        vsm.set_labels(labels)
        res[name] = vsm
        print(MaxPlusMatrixModel().as_dsl(name+'_trace', res))

    # vectortracexml
    if args.operation == OP_MPM_VECTOR_TRACE_XML:

        ni = parse_number_of_iterations(args)
        sequences = parse_sequences(args)

        if len(sequences) ==0:
            # nothing was specified on the command line, use all vector sequences and
            # event sequences as default
            for s in vector_sequences:
                sequences.append(s)
            for s in event_sequences:
                sequences.append(s)
            if len(sequences) == 0:
                # still nothing?
                raise DataflowException("vectortracexml requires sequences.")

        # determine the labels and the final length of the trace
        trace_len = ni
        labels = []
        for s in sequences:
            if not (s in vector_sequences or s in event_sequences):
                raise DataflowException(f"Unknown vector or event sequence {s}.")
            if s in vector_sequences:
                vs = vector_sequences[s]
                for n in range(vs.vector_length()):
                    labels.append(vs.get_label(n, s))
                trace_len = vs.length() if trace_len is None else min(trace_len, vs.length())
            elif s in event_sequences:
                ms = event_sequences[s]
                labels.append(s)
                trace_len = ms.length() if trace_len is None else min(trace_len, ms.length())
            else:
                raise DataflowException(f"Sequence {s} is unknown.")

        # collect the actual trace
        vt = []
        for s in sequences:
            if s in vector_sequences:
                vs = vector_sequences[s]
                for r in mp_transpose_matrix(vs.vectors()):
                    vt.append(r[:trace_len])
            else:
                es = event_sequences[s]
                vt.append(es.sequence()[:trace_len])

        # transpose the result
        vt = mp_transpose_matrix(vt)

        print_xml_trace(fraction_to_float_optional_l_list(vt), labels)

    # convolution
    if args.operation == OP_MPM_CONVOLUTION:
        sequences, res = _convolution(event_sequences, args)
        print(f'The convolution of {", ".join(sequences)} is:')
        print(res)

    # convolutiontransform
    if args.operation == OP_MPM_CONVOLUTION_TRANSFORM:
        sequences, res_s = _convolution(event_sequences, args)
        res = {}
        res[f"{'_'.join(sequences)}_conv"] = res_s
        print(MaxPlusMatrixModel().as_dsl(name+'_conv', res))

    # maxsequences
    if args.operation == OP_MPM_MAXIMUM:
        sequences, res = _maximum(event_sequences, args)
        print(f'The maximum of {", ".join(sequences)} is:')
        print(res)

    # maxsequencestransform
    if args.operation == OP_MPM_MAXIMUM_TRANSFORM:
        sequences, res_s = _maximum(event_sequences, args)
        res = {}
        res[f"{'_'.join(sequences)}_max"] = res_s
        print(MaxPlusMatrixModel().as_dsl(name+'_max', res))

    #'delaysequence'
    if args.operation == OP_MPM_DELAY_SEQUENCE:
        delay = require_parameter_integer(args)
        seq = require_one_event_sequence(event_sequences, args)
        res = event_sequences[seq].delay(delay)
        print(f"The {delay}-delayed sequence of {seq} is:")
        print(res)

    #'scalesequence'
    if args.operation == OP_MPM_SCALE_SEQUENCE:
        scale = require_parameter_mp_value(args)
        seq = require_one_event_sequence(event_sequences, args)
        res = event_sequences[seq].scale(scale)
        print(f"The scaled sequence of {seq} by scaling factor {mp_element_to_string(scale)} is:")
        print(res)



def _determine_input_labels(matrices):
    if len(matrices) == 1:
        return []
    else:
        input_labels, _, _ = determine_state_space_labels(matrices)
        return input_labels


def _make_vector_trace(matrices, vector_sequences, event_sequences, args):
    if len(matrices) == 1:
        ni = require_number_of_iterations(args)
        matrix =matrices.values()[0]
        if not matrix.is_square():
            raise DataflowException("Matrix must be square.")
        x0 = parse_initial_state(args, matrix.numberOfRows())
        vt = matrix.vectorTraceClosed(x0, ni)
        inputs = []
        state_size = matrix.number_of_rows()
        input_labels = []
        output_labels = []
        state_labels = []
    else:
        ni = require_number_of_iterations(args)
        input_labels, state_labels, output_labels = determine_state_space_labels(matrices)
        state_size = len(state_labels)
        x0 = parse_initial_state(args, state_size)
        nt, ut = parse_input_traces(event_sequences, vector_sequences, args)
        inputs = MaxPlusMatrixModel.extract_sequences(nt, ut, event_sequences, \
                                                     vector_sequences, input_labels)
        vt = MaxPlusMatrixModel.vector_trace(matrices, x0, ni, inputs, True)

    labels = []
    labels = labels + input_labels
    labels = labels + state_labels
    labels = labels + output_labels

    return labels, vt

def _convolution(event_sequences, args):
    sequences = parse_sequences(args)
    if len(sequences) < 2:
        raise DataflowException("Please specify at least two sequences to convolve.")
    validate_event_sequences(event_sequences, sequences)

    res = event_sequences[sequences[0]]
    for s in sequences[1:]:
        res = res.convolve_with(event_sequences[s])
    return sequences, res

def _maximum(event_sequences, args):
    sequences = parse_sequences(args)
    if len(sequences) < 2:
        raise DataflowException("Please specify at least two sequences to maximize.")
    validate_event_sequences(event_sequences, sequences)

    res = event_sequences[sequences[0]]
    for s in sequences[1:]:
        res = res.max_with(event_sequences[s])
    return sequences, res

def _determine_trace(dataflow_graph: DataflowGraph, args: Dict[str,Any], ni: int):
    state_size = dataflow_graph.number_of_initial_tokens()
    x0 = parse_initial_state(args, state_size)

    # get input sequences.
    inp_sig = dataflow_graph.input_signals()
    nt, _ = parse_input_traces(inp_sig, {}, args)  # type: ignore type system issue

    if state_size != len(x0):
        raise DataflowException('Initial state vector is of incorrect size.')

    return dataflow_graph.determine_trace(ni, x0, nt)  # type: ignore type system issue


if __name__ == "__main__":
    main()
