""" miscellaneous utility functions """

import re
from typing import Any, Dict, Optional, Tuple, Union, List
from dataflow.maxplus.maxplus import mpParseVector, mpZeroVector, TTimeStamp, TTimeStampList, TMPVector, MP_MINUSINFINITY
from dataflow.libmpm import EventSequenceModel, VectorSequenceModel, MaxPlusMatrixModel

def warn(s: str):
    print("Warning: " + s)

def error(s: str):
    print("Error: "+ s)
    exit()

def makeLabels(name: str, n: int):
    if n == 1:
        return [name]
    return ['{}{}'.format(name, k) for k in range(n)]


def printXmlTrace(vt: List[TTimeStampList], labels: List[str]):
    print('<?xml version="1.0"?>')
    print('<vectortrace>')
    print('    <vectors>')
    k = 0
    vSize = len(labels)
    for v in vt:
        print('        <vector id="{}">'.format(k))
        for n in range(vSize):
            if v[n] is None:
                timestamp  = '-inf'
            else:
                timestamp = v[n]
            print('            <token name="{}" timestamp="{}"/>'.format(labels[n], timestamp))
        print('        </vector>')
        k = k + 1
    print('</vectors>')
    print('</vectortrace>')

def xmlGanttChart(actorNames: List[str], repVec:Dict[str,int], firingStarts: List[TTimeStampList], firingDurations: List[float], inputNames: List[str], inputTraces: List[TTimeStampList], outputNames: List[str], outputTraces: List[TTimeStampList]):
    xml = '<?xml version="1.0"?>\n'
    xml += '<trace>\n'
    xml += '    <firings>\n'
    id = 0
    it = 0
    for v in firingStarts:
        k = 0
        n = 0
        for a in actorNames:
            for _ in range(repVec[a]):
                if v[k] != MP_MINUSINFINITY:
                    vk: float = v[k]  # type: ignore
                    xml += '        <firing id="{}" start="{}" end="{}" actor="{}" iteration="{}" scenario="s"/>\n'.format(id, vk, vk+firingDurations[n], a, it)
                    id += 1
                k += 1
            n += 1
        it += 1
    xml += '    </firings>\n'
    xml += '    <inputs>\n'
    it = 0
    for v in inputTraces:
        k = 0
        for i in inputNames:
            for _ in range(repVec[i]):
                if v[k] != MP_MINUSINFINITY:
                    xml += '        <input name="{}" timestamp="{}" iteration="{}"/>\n'.format(i, v[k], it)
                    it += 1
                k += 1

    xml += '    </inputs>\n'
    xml += '    <outputs>\n'
    it = 0
    for v in outputTraces:
        k = 0
        for o in outputNames:
            for _ in range(repVec[o]):
                if v[k] != MP_MINUSINFINITY:
                    xml += '        <output name="{}" timestamp="{}" iteration="{}"/>\n'.format(o, v[k], it)
                    it += 1
                k += 1

    xml += '    </outputs>\n'
    xml += '</trace>\n'
    return xml


def printXmlGanttChart(actorNames: List[str], repVec:Dict[str,int], firingStarts: List[TTimeStampList], firingDurations: List[float], inputNames: List[str], inputTraces: List[TTimeStampList], outputNames: List[str], outputTraces: List[TTimeStampList]):
    print(xmlGanttChart(actorNames, repVec, firingStarts, firingDurations, inputNames, inputTraces, outputNames, outputTraces))


def inputTracesRegEx():
    '''
    syntax for inputtrace: comma-separated list of: (ID=)?([...]|ID)
    '''
    return r"(([a-zA-Z][a-zA-Z0-9_]*=)?((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*)))(,(([a-zA-Z][a-zA-Z0-9_]*=)?((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*))))*"

def inputTraceRegEx():
    '''
    syntax for inputtrace: (ID=)?([...]|ID)
    '''
    return r"^(([a-zA-Z][a-zA-Z0-9_]*=)?((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*)))"

def namedInputTraceRegEx():
    '''
    syntax for named inputtrace: ID=([...]|ID)
    '''
    return r"([a-zA-Z][a-zA-Z0-9_]*)=((\[.*?\])|([a-zA-Z][a-zA-Z0-9_]*))"

def sequenceLiteralRegEx():
    '''
    syntax for a literal sequence: [...]
    '''
    return r"\[.*?\]"

def parseInputTraces(eventsequences: Dict[str,Union[TTimeStampList,EventSequenceModel]], vectorsequences: Dict[str,VectorSequenceModel], args: Any) -> Union[Tuple[None,None],Tuple[Dict[str,TTimeStampList],List[TTimeStampList]]]:
    '''
    return dictionary of named traces and list of unnamed traces
    '''

    # check if trace are specified 
    if not args.inputtrace:
        return None, None

    it = args.inputtrace

    # check if they are syntactically correct
    if not re.match(inputTracesRegEx() , it):
        raise Exception("Invalid input trace specification")

    # variables to collect results for named and unnamed traces
    resNt: Dict[str,TTimeStampList] = dict()
    resUt:List[TTimeStampList] = list()
    
    # find individual traces
    traces: List[str] = []
    ss = re.search(inputTraceRegEx(), it)
    while ss:
        tt:str = ss.group(1)
        traces.append(tt)
        it = it[len(tt):]
        if len(tt)>0:
            it = it[1:]
        ss = re.search(inputTraceRegEx(), it)

    # for each trace found
    for t in traces:
        # check if it is a named trace, e.g., x=[1,2,3]
        namedMatch = re.match(namedInputTraceRegEx(), t)
        if namedMatch:
            # it is a named trace
            name = namedMatch.group(1)
            expr = namedMatch.group(2)
            if re.match(sequenceLiteralRegEx(), expr):
                resNt[name] = mpParseVector(expr)
            else:
                if expr in eventsequences:
                    # ugly, but if the model is SDF, this is a list, if the model is max-plus this is an EventSequenceModel
                    evSeqOrList = eventsequences[expr]
                    if isinstance(evSeqOrList, list):
                        resNt[name] = evSeqOrList
                    else:
                        resNt[name] = evSeqOrList.sequence()
                elif expr in vectorsequences:
                    vs = vectorsequences[expr]
                    vsl = vs.extractSequenceList()
                    for k in range(len(vs)):
                        resNt[name+str(k+1)] = vsl[k]
                else:
                    raise Exception("Unknown sequence: {}".format(expr))
        else:
            # check if it is a literal, unnamed, event sequence, e.g. [1,2,3]
            if re.match(sequenceLiteralRegEx(), t):
                resUt.append(mpParseVector(t))
            else:
                if t in eventsequences:
                    es = eventsequences[t]
                    if isinstance(es, EventSequenceModel):
                        resUt.append(es.sequence())
                    else:
                        resUt.append(es)
                elif t in vectorsequences:
                    vs = vectorsequences[t]
                    vsl = vs.extractSequenceList()
                    for k in range(len(vs)):
                        resUt.append(vsl[k])
                else:
                    raise Exception("Unknown sequence: {}".format(t))

    return resNt, resUt

def parsePeriod(args: Any)->Optional[float]:
    if not args.period:
        return None
    try:
        return float(args.period)
    except Exception:
        raise Exception("Failed to parse period argument; period must be a floating point number.")

def requirePeriod(args)->float:
    if not args.period:
        raise Exception("Operation requires period to be given.")
    val = parsePeriod(args)
    if val is None:
        raise Exception("Operation requires period to be given.")
    return val

def parseParameterInteger(args: Any)->Optional[int]:
    if not args.parameter:
        return None
    try:
        return int(args.parameter)
    except Exception:
        raise Exception("Failed to parse integer parameter.")

def parseParameterMPValue(args: Any)->Tuple[bool, TTimeStamp]:
    '''Returns False, None if parameter was not specified, otherwise return True, value. An exception is raised if the parsing failed.'''
    if not args.parameter:
        return False, None
    if args.parameter == 'mininf':
        return True, MP_MINUSINFINITY
    try:
        return True, float(args.parameter)
    except Exception:
        raise Exception("Failed to parse parameter as a max-plus value.")

def requireParameterInteger(args) -> int:
    val = parseParameterInteger(args)
    if val is None:
        raise Exception("Operation requires parameter to be specified as an integer.")
    return val

def requireParameterMPValue(args)->TTimeStamp:
    '''Parse and return required MP parameter value. If it doesn't exist, or cannot be parsed, an exception is raised.'''
    if not args.parameter:
        raise Exception("Operation requires parameter to be specified as a floating point number or 'mininf'.")
    success, val =  parseParameterMPValue(args)
    if not success:
        raise Exception("Operation requires parameter to be specified as a floating point number or 'mininf'.")
    return val

def parseInitialState(args: Any, stateSize: int) -> TMPVector:
    if args.initialstate is None:
        return mpZeroVector(stateSize)
    else:
        x0 = mpParseVector(args.initialstate)
        if len(x0) != stateSize:
            raise Exception('Provided initial state is not of the expected size.')
        return x0

def requireNumberOfIterations(args: Any) -> int:
    if not args.numberofiterations:
        raise Exception("Operation requires number of iterations to be given.")
    val = parseNumberOfIterations(args)
    if val is None:
        raise Exception("Failed to parse number of iterations.")
    return val


def parseNumberOfIterations(args: Any) -> Optional[int]:
    if not args.numberofiterations:
        return None
    try:
        return int(args.numberofiterations)
    except Exception:
        raise Exception("Failed to parse numberofiterations argument; it must be a non-negative integer number.")
    
def parseMatrices(args: Any)->List[str]:
    '''Parse list of matrix names.'''
    if not args.matrices:
        return []
    return args.matrices.split(',')

def requireMatrices(argMatrices: List[str], defMatrices: Dict[str, MaxPlusMatrixModel])->List[str]:
    ''' check that the matrices in the list argMatrices are all defined in defMatrices, and that at least one matrix is specified, otherwise raise an exception. Returns argMatrices.'''
    if len(argMatrices) == 0:
        raise Exception("One or more matrices must be specified.")
    for m in argMatrices:
        if not m in defMatrices:
            raise Exception("Matrix {0} is not defined in the model.".format(m))
    return argMatrices

def parseOneMatrix(args: Any)->Optional[str]:
    '''Get one matrix name from the arguments, or None if there is no such argument'''
    matrices = parseMatrices(args)
    if len(matrices) == 0:
        return None
    if len(matrices) != 1:
        raise Exception("Specify one matrix only.")
    return matrices[0]

def requireSquareMatrices(matrices: Dict[str, MaxPlusMatrixModel])->List[str]:
    '''Return a list of names of the square matrices in the matrices dictionary. If there are no square matrices, an exception is raised.'''
    sqMatrixModels = [m for m,M in matrices.items() if M.isSquare()] 
    if len(sqMatrixModels) == 0:
        raise Exception("Provide a model with a square matrix.")
    return sqMatrixModels

def getSquareMatrix(matrices: Dict[str, MaxPlusMatrixModel], args: Any) -> str:
    '''Get a square matrix from the arguments. Requires that exactly one, square matrix was specified, otherwise an exception is raised.'''
    sqMatrixModels = requireSquareMatrices(matrices)
    mat = parseOneMatrix(args)
    if mat is None:
        raise Exception("No matrix was specified.")
    if mat:
        if mat not in sqMatrixModels:
            raise Exception("Matrix {0} is not square.".format(mat))
    else:
        mat = next(iter(sqMatrixModels))
    return mat

def parseSequences(args: Any) -> List[str]:
    '''Return a list of sequence names specified in arg.'''
    if not args.sequences:
        return []
    return args.sequences.split(',')

def validateEventSequences(eventsequences: Dict[str,object], sequences: List[str]):
    ''' check that the sequences are defined in the eventsequences dict.'''
    for s in sequences:
        if s not in eventsequences:
            raise Exception("Sequence {} is not defined.".format(s))


def parseOneSequence(args: Any) -> Optional[str]:
    '''Return the name of a specified sequence. If the argument does not specify sequences, None i returned. If multiple sequences are specified an exception is raised.'''
    sequences = parseSequences(args)
    if len(sequences) == 0:
        return None
    if len(sequences) != 1:
        raise Exception("Specify one sequence.")
    return sequences[0]


def requireOneEventSequence(eventsequences: Dict[str,Any], args)->str:
    '''Return the name of a specified sequence. Requires that exactly on sequence is specified from eventsequences, otherwise an exception is raised.'''
    s = parseOneSequence(args)
    if s is None:
        raise Exception("A sequence is required.")
    if not s in eventsequences:
        raise Exception("Sequence {} is unknown.".format(s))
    return s

def requireSequenceOfMatricesAndPossiblyVectorSequence(matrices:Dict[str,MaxPlusMatrixModel], vectorsequences: Dict[str,VectorSequenceModel], args: Any) -> List[str]:
    '''Get a required sequence of matrices, and possibly last a vector sequence. An exception is raised if no sequence is specified, or it is malformed.'''
    names = requireMatrices(parseMatrices(args), matrices)
    for k in range(len(names)):
        if names[k] not in matrices:
            raise Exception('{} is not a matrix'.format(names[k]))
    vs = parseOneSequence(args)
    if vs:
        if vs not in vectorsequences:
            raise Exception('{} is not a vector sequence'.format(vs))
        names.append(vs)

    return names

def requireMatrixDefined(matrices: Dict[str,MaxPlusMatrixModel], m: str)->MaxPlusMatrixModel:
    '''Check that m is defined in matrices and return the corresponding matrix model. If m is not defined, raises an exception'''
    if not m in matrices:
        raise Exception("Matrix {} is not defined.".format(m))
    return matrices[m]

def determineStateSpaceLabels(matrices: Dict[str,MaxPlusMatrixModel])->Tuple[List[str],List[str],List[str]]:
    '''Determine the labels of the state-space elements from the matrix definitions. Returns a tuple with input labels, state labels, and output labels.'''
    MA = requireMatrixDefined(matrices, "A")
    MB = requireMatrixDefined(matrices, "B")
    MC = requireMatrixDefined(matrices, "C")
    requireMatrixDefined(matrices, "D")
    inputSize = MB.numberOfColumns()
    stateSize = MA.numberOfRows()
    outputSize = MC.numberOfRows()
    if len(MB.labels()) == stateSize+inputSize:
        inputLabels = (MB.labels())[stateSize:]
    else:
        # make default input names
        inputLabels = makeLabels('i', inputSize)

    if len(MA.labels()) >= stateSize:
        stateLabels = (MA.labels())[:stateSize]
    else:
        # make default state labels
        stateLabels = makeLabels('x', stateSize)

    if len(MC.labels()) >= outputSize:
        outputLabels = (MC.labels())[:outputSize]
    else:
        # make default output labels
        outputLabels = makeLabels('o', outputSize)
    return inputLabels, stateLabels, outputLabels