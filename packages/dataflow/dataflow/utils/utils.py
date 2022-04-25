""" miscellaneous utility functions """

import re
from dataflow.maxplus.maxplus import mpParseVector, MP_MINUSINFINITY
from dataflow.maxplus.maxplus import mpZeroVector

def warn(s):
    print("Warning: " + s)

def error(s):
    print("Error: "+ s)
    exit()

def makeLabels(name, n):
    if n == 1:
        return [name]
    return ['{}{}'.format(name, k) for k in range(n)]


def printXmlTrace(vt, labels):
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

def xmlGanttChart(actorNames, repVec, firingStarts, firingDurations, inputNames, inputTraces, outputNames, outputTraces):
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
                    xml += '        <firing id="{}" start="{}" end="{}" actor="{}" iteration="{}" scenario="s"/>\n'.format(id, v[k], v[k]+firingDurations[n], a, it)
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


def printXmlGanttChart(actorNames, repVec, firingStarts, firingDurations, inputNames, inputTraces, outputNames, outputTraces):
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

def parseInputTraces(eventsequences, vectorsequences, args):
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
    resNt = dict()
    resUt = list()
    
    # find individual traces
    traces = []
    ss = re.search(inputTraceRegEx(), it)
    while ss:
        tt = ss.group(1)
        traces.append(tt)
        it = it[len(tt):]
        if len(tt)>0:
            it = it[1:]
        ss = re.search(inputTraceRegEx(), it)

    # for each trace found
    for t in traces:
        namedMatch = re.match(namedInputTraceRegEx(), t)
        if namedMatch:
            # it is a named trace
            name = namedMatch.group(1)
            expr = namedMatch.group(2)
            if re.match(sequenceLiteralRegEx(), expr):
                resNt[name] = mpParseVector(expr)
            else:
                if expr in eventsequences:
                    # ugly, but if the model is SDF, this is a list, it the model is max-plus this is an EventSequenceModel
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
            if re.match(sequenceLiteralRegEx(), t):
                resUt.append(mpParseVector(t))
            else:
                if t in eventsequences:
                    resUt.append(eventsequences[t].sequence())
                elif t in vectorsequences:
                    vs = vectorsequences[t]
                    vsl = vs.extractSequenceList()
                    for k in range(len(vs)):
                        resUt.append(vsl[k])
                else:
                    raise Exception("Unknown sequence: {}".format(t))

    return resNt, resUt

def parsePeriod(args):
    if not args.period:
        return None
    try:
        return float(args.period)
    except Exception:
        raise Exception("Failed to parse period argument; period must be a floating point number.")

def requirePeriod(args):
    if not args.period:
        raise Exception("Operation requires period to be given.")
    return parsePeriod(args)

def parseParameterInteger(args):
    if not args.parameter:
        return None
    try:
        return int(args.parameter)
    except Exception:
        raise Exception("Failed to parse integer parameter.")

def parseParameterMPValue(args):
    if not args.parameter:
        return None
    if args.parameter == 'mininf':
        return MP_MINUSINFINITY
    try:
        return float(args.parameter)
    except Exception:
        raise Exception("Failed to parse parameter as a max-plus value.")

def requireParameterInteger(args):
    if not args.parameter:
        raise Exception("Operation requires parameter to be specified as an integer.")
    return parseParameterInteger(args)

def requireParameterMPValue(args):
    if not args.parameter:
        raise Exception("Operation requires parameter to be specified as a floating point number or 'mininf'.")
    return parseParameterMPValue(args)

def parseInitialState(args, stateSize):
    if args.initialstate is None:
        return mpZeroVector(stateSize)
    else:
        x0 = mpParseVector(args.initialstate)
        if len(x0) != stateSize:
            raise Exception('Provided initial state is not of the expected size.')
        return x0

def requireNumberOfIterations(args):
    if not args.numberofiterations:
        raise Exception("Operation requires number of iterations to be given.")
    return parseNumberOfIterations(args)

def parseNumberOfIterations(args):
    if not args.numberofiterations:
        return None
    try:
        return int(args.numberofiterations)
    except Exception:
        raise Exception("Failed to parse numberofiterations argument; it must be a non-negative integer number.")
    
def parseMatrices(args):
    if not args.matrices:
        return []
    return args.matrices.split(',')

def requireMatrices(argMatrices, defMatrices):
    # check that the matrices in the list argMatrices are all defined in defMatrices, and that at least one matrix is specified, otherwise raise an exception
    # returns argMatrices
    if len(argMatrices) == 0:
        raise Exception("One or more matrices must be specified.")
    for m in argMatrices:
        if not m in defMatrices:
            raise Exception("Matrix {0} is not defined in the model.".format(m))
    return argMatrices

def parseOneMatrix(args):
    matrices = parseMatrices(args)
    if len(matrices) == 0:
        return None
    if len(matrices) != 1:
        raise Exception("Specify one matrix.")
    return matrices[0]

def requireSquareMatrices(matrices):
    sqMatrixModels = [m for m,M in matrices.items() if M.isSquare()] 
    if len(sqMatrixModels) == 0:
        raise Exception("Provide a model with a square matrix.")
    return sqMatrixModels

def getSquareMatrix(matrices, args):
    sqMatrixModels = requireSquareMatrices(matrices)
    mat = parseOneMatrix(args)
    if mat:
        if mat not in sqMatrixModels:
            raise Exception("Matrix {0} is not square.".format(mat))
    else:
        mat = next(iter(sqMatrixModels))
    return mat

def parseSequences(args):
    if not args.sequences:
        return []
    return args.sequences.split(',')

def validateEventSequences(eventsequences, sequences):
    # check that the sequences are defined
    for s in sequences:
        if s not in eventsequences:
            raise Exception("Sequence {} is not defined.".format(s))


def parseOneSequence(args):
    sequences = parseSequences(args)
    if len(sequences) == 0:
        return None
    if len(sequences) != 1:
        raise Exception("Specify one sequence.")
    return sequences[0]


def requireOneEventSequence(eventsequences, args):
    s = parseOneSequence(args)
    if not s:
        raise Exception("A sequence is required.")
    if not s in eventsequences:
        raise Exception("Sequence {} is unknown.".format(s))
    return s

def requireSequenceOfMatricesAndPossiblyVectorSequence(matrices, vectorsequences, args):
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

def requireMatrixDefined(matrices, m):
    if not m in matrices:
        raise Exception("Matrix {} is not defined.".format(m))
    return matrices[m]

def determineStateSpaceLabels(matrices):
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
        inputLabels = makeLabels('i', inputSize)

    if len(MA.labels()) >= stateSize:
        stateLabels = (MA.labels())[:stateSize]
    else:
        stateLabels = makeLabels('x', stateSize)

    if len(MC.labels()) >= outputSize:
        outputLabels = (MC.labels())[:outputSize]
    else:
        outputLabels = makeLabels('o', outputSize)
    return inputLabels, stateLabels, outputLabels