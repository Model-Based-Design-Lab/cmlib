""" miscellaneous utility functions """

import re
from fractions import Fraction
import sys
from typing import Any, Dict, Optional, Tuple, Union, List
from dataflow.maxplus.maxplus import mpParseVector, mpZeroVector
from dataflow.maxplus.algebra import MP_MINUSINFINITY_STR, MP_MINUSINFINITY
from dataflow.maxplus.types import TMPVector, TTimeStamp, TTimeStampList, TTimeStampFloatList
from dataflow.libmpm import EventSequenceModel, VectorSequenceModel, MaxPlusMatrixModel

class DataflowException(Exception):
    """Exceptions related to this package"""

def warn(s: str):
    """Print warning."""
    print("Warning: " + s)

def error(s: str):
    """Print error."""
    print("Error: "+ s)
    sys.exit()

def make_labels(name: str, n: int):
    """Make labels from name and number."""
    if n == 1:
        return [name]
    return [f'{name}{k}' for k in range(n)]


def print_xml_trace(vt: List[TTimeStampFloatList], labels: List[str]):
    """Print an xml representation of the trace using the labels."""
    print('<?xml version="1.0"?>')
    print('<vectortrace>')
    print('    <vectors>')
    k = 0
    v_size = len(labels)
    for v in vt:
        print(f'        <vector id="{k}">')
        for n in range(v_size):
            if v[n] is None:
                timestamp  = MP_MINUSINFINITY_STR
            else:
                timestamp = v[n]
            print(f'            <token name="{labels[n]}" timestamp="{timestamp}"/>')
        print('        </vector>')
        k = k + 1
    print('</vectors>')
    print('</vectortrace>')

def xml_gantt_chart(actor_names: List[str], rep_vec:Dict[str,int], firing_starts: List[TTimeStampFloatList], firing_durations: List[float], input_names: List[str], input_traces: List[TTimeStampFloatList], output_names: List[str], output_traces: List[TTimeStampFloatList]):
    """Make an XML Gantt chart representation."""
    xml = '<?xml version="1.0"?>\n'
    xml += '<trace>\n'
    xml += '    <firings>\n'
    f_id = 0
    it = 0
    for v in firing_starts:
        k = 0
        n = 0
        for a in actor_names:
            for _ in range(rep_vec[a]):
                if v[k] != MP_MINUSINFINITY:
                    vk: float = v[k]  # type: ignore
                    xml += f'        <firing id="{f_id}" start="{vk}" end="{vk+firing_durations[n]}" actor="{a}" iteration="{it}" scenario="s"/>\n'
                    f_id += 1
                k += 1
            n += 1
        it += 1
    xml += '    </firings>\n'
    xml += '    <inputs>\n'
    it = 0
    for v in input_traces:
        k = 0
        for i in input_names:
            for _ in range(rep_vec[i]):
                if v[k] != MP_MINUSINFINITY:
                    xml += f'        <input name="{i}" timestamp="{v[k]}" iteration="{it}"/>\n'
                    it += 1
                k += 1

    xml += '    </inputs>\n'
    xml += '    <outputs>\n'
    it = 0
    for v in output_traces:
        k = 0
        for o in output_names:
            for _ in range(rep_vec[o]):
                if v[k] != MP_MINUSINFINITY:
                    xml += f'        <output name="{o}" timestamp="{v[k]}" iteration="{it}"/>\n'
                    it += 1
                k += 1

    xml += '    </outputs>\n'
    xml += '</trace>\n'
    return xml


def print_xml_gantt_chart(actor_names: List[str], rep_vec:Dict[str,int], firing_starts: List[TTimeStampFloatList], firing_durations: List[float], input_names: List[str], input_traces: List[TTimeStampFloatList], output_names: List[str], output_traces: List[TTimeStampFloatList]):
    """Print an XML representation of the Gantt chart."""
    print(xml_gantt_chart(actor_names, rep_vec, firing_starts, firing_durations, \
                          input_names, input_traces, output_names, output_traces))


def input_traces_reg_ex():
    '''
    syntax for inputtrace: comma-separated list of: (ID=)?([...]|ID)
    '''
    return r"(([a-zA-Z][a-zA-Z0-9_]*=)?((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*)))(,(([a-zA-Z][a-zA-Z0-9_]*=)?((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*))))*"

def input_trace_reg_ex():
    '''
    syntax for inputtrace: (ID=)?([...]|ID)
    '''
    return r"^(([a-zA-Z][a-zA-Z0-9_]*=)?((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*)))"

def named_input_trace_reg_ex():
    '''
    syntax for named inputtrace: ID=([...]|ID)
    '''
    return r"([a-zA-Z][a-zA-Z0-9_]*)=((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*))"

def sequence_literal_reg_ex():
    '''
    syntax for a literal sequence: [...]
    '''
    return r"\[.*?\]"

def parse_input_traces(eventsequences: Dict[str,Union[TTimeStampList,EventSequenceModel]], vectorsequences: Dict[str,VectorSequenceModel], args: Any) -> Union[Tuple[None,None],Tuple[Dict[str,TTimeStampList],List[TTimeStampList]]]:
    '''
    return dictionary of named traces and list of unnamed traces
    '''

    # check if trace are specified
    if not args.inputtrace:
        return None, None

    it = args.inputtrace

    # check if they are syntactically correct
    if not re.match(input_traces_reg_ex() , it):
        raise DataflowException("Invalid input trace specification")

    # variables to collect results for named and unnamed traces
    res_nt: Dict[str,TTimeStampList] = dict()
    res_ut:List[TTimeStampList] = list()

    # find individual traces
    traces: List[str] = []
    ss = re.search(input_trace_reg_ex(), it)
    while ss:
        tt:str = ss.group(1)
        traces.append(tt)
        it = it[len(tt):]
        if len(tt)>0:
            it = it[1:]
        ss = re.search(input_trace_reg_ex(), it)

    # for each trace found
    for t in traces:
        # check if it is a named trace, e.g., x=[1,2,3]
        named_match = re.match(named_input_trace_reg_ex(), t)
        if named_match:
            # it is a named trace
            name = named_match.group(1)
            expr = named_match.group(2)
            if re.match(sequence_literal_reg_ex(), expr):
                res_nt[name] = mpParseVector(expr)
            else:
                if expr in eventsequences:
                    # ugly, but if the model is SDF, this is a list, if the model
                    # is max-plus this is an EventSequenceModel
                    ev_seq_or_list = eventsequences[expr]
                    if isinstance(ev_seq_or_list, list):
                        res_nt[name] = ev_seq_or_list
                    else:
                        res_nt[name] = ev_seq_or_list.sequence()
                elif expr in vectorsequences:
                    vs = vectorsequences[expr]
                    vsl = vs.extract_sequence_list()
                    for k in range(len(vs)):
                        res_nt[name+str(k+1)] = vsl[k]
                else:
                    raise DataflowException(f"Unknown sequence: {expr}")
        else:
            # check if it is a literal, unnamed, event sequence, e.g. [1,2,3]
            if re.match(sequence_literal_reg_ex(), t):
                res_ut.append(mpParseVector(t))
            else:
                if t in eventsequences:
                    es = eventsequences[t]
                    if isinstance(es, EventSequenceModel):
                        res_ut.append(es.sequence())
                    else:
                        res_ut.append(es)
                elif t in vectorsequences:
                    vs = vectorsequences[t]
                    vsl = vs.extract_sequence_list()
                    for k in range(len(vs)):
                        res_ut.append(vsl[k])
                else:
                    raise DataflowException(f"Unknown sequence: {t}")

    return res_nt, res_ut

def parse_period(args: Any)->Optional[Fraction]:
    """Parse specified period argument."""
    if not args.period:
        return None
    try:
        return Fraction(args.period)
    except Exception:
        raise DataflowException("Failed to parse period argument; period must be a "\
                                "floating point number.") # pylint: disable=raise-missing-from

def require_period(args)->Fraction:
    """Ensure that a period has been provided."""
    if not args.period:
        raise DataflowException("Operation requires period to be given.")
    val = parse_period(args)
    if val is None:
        raise DataflowException("Operation requires period to be given.")
    return val

def parse_parameter_integer(args: Any)->Optional[int]:
    """Parse parameter as an integer."""
    if not args.parameter:
        return None
    try:
        return int(args.parameter)
    except Exception:
        raise DataflowException("Failed to parse integer parameter.") # pylint: disable=raise-missing-from

def parse_parameter_mp_value(args: Any)->Tuple[bool, TTimeStamp]:
    '''Returns False, None if parameter was not specified, otherwise return True,
    value. An exception is raised if the parsing failed.'''
    if not args.parameter:
        return False, None
    if args.parameter == 'mininf':
        return True, MP_MINUSINFINITY
    try:
        return True, Fraction(float(args.parameter))
    except Exception:
        raise DataflowException("Failed to parse parameter as a max-plus value.") # pylint: disable=raise-missing-from

def require_parameter_integer(args) -> int:
    """Ensure parameter is specified and parse it."""
    val = parse_parameter_integer(args)
    if val is None:
        raise DataflowException("Operation requires parameter to be specified as an integer.") # pylint: disable=raise-missing-from
    return val

def require_parameter_mp_value(args)->TTimeStamp:
    '''Parse and return required MP parameter value. If it doesn't exist, or cannot
    be parsed, an exception is raised.'''
    if not args.parameter:
        raise DataflowException("Operation requires parameter to be specified as a "\
                                "floating point number or 'mininf'.") # pylint: disable=raise-missing-from
    success, val =  parse_parameter_mp_value(args)
    if not success:
        raise DataflowException("Operation requires parameter to be specified as a " \
                                "floating point number or 'mininf'.") # pylint: disable=raise-missing-from
    return val

def parse_initial_state(args: Any, state_size: int) -> TMPVector:
    """Parse initial state."""
    if args.initialstate is None:
        return mpZeroVector(state_size)
    x0 = mpParseVector(args.initialstate)
    if len(x0) != state_size:
        raise DataflowException('Provided initial state is not of the expected size.') # pylint: disable=raise-missing-from
    return x0

def require_number_of_iterations(args: Any) -> int:
    """Ensure that number of iterations is specified and parse it."""
    if not args.numberofiterations:
        raise DataflowException("Operation requires number of iterations to be given.") # pylint: disable=raise-missing-from
    val = parse_number_of_iterations(args)
    if val is None:
        raise DataflowException("Failed to parse number of iterations.") # pylint: disable=raise-missing-from
    return val


def parse_number_of_iterations(args: Any) -> Optional[int]:
    """Parse number of iterations."""
    if not args.numberofiterations:
        return None
    try:
        return int(args.numberofiterations)
    except Exception:
        raise DataflowException("Failed to parse numberofiterations argument; it must be " \
                                "a non-negative integer number.") # pylint: disable=raise-missing-from

def parse_matrices(args: Any)->List[str]:
    '''Parse list of matrix names.'''
    if not args.matrices:
        return []
    return args.matrices.split(',')

def require_matrices(arg_matrices: List[str], def_matrices: Dict[str, MaxPlusMatrixModel])\
    ->List[str]:
    ''' check that the matrices in the list argMatrices are all defined in defMatrices,
    and that at least one matrix is specified, otherwise raise an exception. Returns argMatrices.'''
    if len(arg_matrices) == 0:
        raise DataflowException("One or more matrices must be specified.")
    for m in arg_matrices:
        if not m in def_matrices:
            raise DataflowException(f"Matrix {m} is not defined in the model.")
    return arg_matrices

def parse_one_matrix(args: Any)->Optional[str]:
    '''Get one matrix name from the arguments, or None if there is no such argument'''
    matrices = parse_matrices(args)
    if len(matrices) == 0:
        return None
    if len(matrices) != 1:
        raise DataflowException("Specify one matrix only.")
    return matrices[0]

def require_square_matrices(matrices: Dict[str, MaxPlusMatrixModel])->List[str]:
    '''Return a list of names of the square matrices in the matrices dictionary.
    If there are no square matrices, an exception is raised.'''
    sq_matrix_models = [m for m,M in matrices.items() if M.is_square()]
    if len(sq_matrix_models) == 0:
        raise DataflowException("Provide a model with a square matrix.")
    return sq_matrix_models

def get_square_matrix(matrices: Dict[str, MaxPlusMatrixModel], args: Any) -> str:
    '''Get a square matrix from the arguments. If no square matrix was specified,
    take the first one from the model.'''
    sq_matrix_models = require_square_matrices(matrices)
    mat = parse_one_matrix(args)
    if mat:
        if mat not in sq_matrix_models:
            raise DataflowException(f"Matrix {mat} is not square.")
    else:
        mat = next(iter(sq_matrix_models))
    return mat

def parse_sequences(args: Any) -> List[str]:
    '''Return a list of sequence names specified in arg.'''
    if not args.sequences:
        return []
    return args.sequences.split(',')

def validate_event_sequences(eventsequences: Dict[str,object], sequences: List[str]):
    ''' check that the sequences are defined in the eventsequences dict.'''
    for s in sequences:
        if s not in eventsequences:
            raise DataflowException(f"Sequence {s} is not defined.")


def parse_one_sequence(args: Any) -> Optional[str]:
    '''Return the name of a specified sequence. If the argument does
    not specify sequences, None i returned. If multiple sequences are
    specified an exception is raised.'''
    sequences = parse_sequences(args)
    if len(sequences) == 0:
        return None
    if len(sequences) != 1:
        raise DataflowException("Specify one sequence.")
    return sequences[0]


def require_one_event_sequence(eventsequences: Dict[str,Any], args)->str:
    '''Return the name of a specified sequence. Requires that exactly on
    sequence is specified from eventsequences, otherwise an exception is raised.'''
    s = parse_one_sequence(args)
    if s is None:
        raise DataflowException("A sequence is required.")
    if not s in eventsequences:
        raise DataflowException(f"Sequence {s} is unknown.")
    return s

def require_sequence_of_matrices_and_possibly_vector_sequence(matrices:Dict[str,\
                MaxPlusMatrixModel], vectorsequences: Dict[str,VectorSequenceModel], args: Any)\
                 -> List[str]:
    '''Get a required sequence of matrices, and possibly last a vector sequence.
    An exception is raised if no sequence is specified, or it is malformed.'''
    names = require_matrices(parse_matrices(args), matrices)
    for n in names:
        if n not in matrices:
            raise DataflowException(f'{n} is not a matrix')
    vs = parse_one_sequence(args)
    if vs:
        if vs not in vectorsequences:
            raise DataflowException(f'{vs} is not a vector sequence')
        names.append(vs)

    return names

def require_matrix_defined(matrices: Dict[str,MaxPlusMatrixModel], m: str)->MaxPlusMatrixModel:
    '''Check that m is defined in matrices and return the corresponding matrix model.
    If m is not defined, raises an exception'''
    if not m in matrices:
        raise DataflowException(f"Matrix {m} is not defined.")
    return matrices[m]

def determine_state_space_labels(matrices: Dict[str,MaxPlusMatrixModel])->\
    Tuple[List[str],List[str],List[str]]:
    '''Determine the labels of the state-space elements from the matrix definitions.
    Returns a tuple with input labels, state labels, and output labels.'''
    matrix_a = require_matrix_defined(matrices, "A")
    matrix_b = require_matrix_defined(matrices, "B")
    matrix_c = require_matrix_defined(matrices, "C")
    require_matrix_defined(matrices, "D")
    input_size = matrix_b.number_of_columns()
    state_size = matrix_a.number_of_rows()
    output_size = matrix_c.number_of_rows()
    if len(matrix_b.labels()) == state_size+input_size:
        input_labels = (matrix_b.labels())[state_size:]
    else:
        # make default input names
        input_labels = make_labels('i', input_size)

    if len(matrix_a.labels()) >= state_size:
        state_labels = (matrix_a.labels())[:state_size]
    else:
        # make default state labels
        state_labels = make_labels('x', state_size)

    if len(matrix_c.labels()) >= output_size:
        output_labels = (matrix_c.labels())[:output_size]
    else:
        # make default output labels
        output_labels = make_labels('o', output_size)
    return input_labels, state_labels, output_labels

def fraction_to_float_list(l: List[Fraction])->List[float]:
    """Convert a list of fractions to a list of floats."""
    return [float(f) for f in l]

def fraction_to_float_optional_list(l: List[Union[Fraction,None]])->List[Union[float,None]]:
    """Convert a list of optional fractions to a list of optional floats."""
    return [None if f is None else float(f) for f in l]

def fraction_to_float_l_list(l: List[List[Fraction]])->List[List[float]]:
    """Convert a list of lists of fractions to a list of lists of floats."""
    return [fraction_to_float_list(ll) for ll in l]

def fraction_to_float_optional_l_list(l: List[List[Union[Fraction,None]]])-> \
    List[List[Union[float,None]]]:
    """Convert a list of lists of optional fractions to a list of lists of optional floats."""
    return [fraction_to_float_optional_list(ll) for ll in l]
