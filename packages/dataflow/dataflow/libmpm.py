from functools import reduce
from io import StringIO
from dataflow.libmpmgrammar import parseMPMDSL
import dataflow.maxplus.maxplus as mp
from dataflow.maxplus.starclosure import starClosure, PositiveCycleException


class EventSequenceModel(object):

    def __init__(self, seq = []):
        self._sequence = seq
    
    def validate(self):
        pass

    def addRow(self, row):
        self._sequence = row

    def sequence(self):
        return self._sequence

    def length(self):
        return len(self._sequence)

    def convolveWith(self, es):
        res = EventSequenceModel()
        res._sequence = mp.mpConvolution(self._sequence, es._sequence)
        return res

    def maxWith(self, es):
        if not isinstance(es, EventSequenceModel):
            raise Exception('Object of the wrong type to take maximum with event sequence.')
        res = EventSequenceModel()
        res._sequence = mp.mpMaxEventSequences(self._sequence, es._sequence)
        return res

    def delay(self, n):
        res = EventSequenceModel()
        res._sequence = mp.mpDelay(self._sequence, n)
        return res

    def scale(self, c):
        res = EventSequenceModel()
        res._sequence = mp.mpScale(self._sequence, c)
        return res

    def extractSequenceList(self):
        return [self._sequence]

    def __str__(self):
        return mp.mpVector(self._sequence)

class VectorSequenceModel(object):

    def __init__(self, vectors=None):
        if vectors is None:
            self._vectors = []
        else:
            self._vectors = vectors
        self._labels = []

    def addVector(self, v):
        self.addRow(v)

    def addRow(self, row):
        self._vectors.append(row)

    def setLabels(self, labels):
        self._labels = labels

    def labels(self):
        return self._labels

    def getLabel(self, n, base = 's'):
        if n < len(self._labels):
            return self._labels[n]
        else:
            return base + (str(n+1))

    def vectors(self):
        return self._vectors

    def length(self):
        return len(self._vectors)

    def vectorLength(self):
        if len(self._vectors) == 0:
            return 0
        else:
            return len(self._vectors[0])

    def validate(self):
        # check if all vectors are of the same length
        if len(set([len(r) for r in self._vectors])) > 1:
            raise Exception("Vector sequence contains vectors of unequal length.")

    def maxWith(self, vs):
        res = VectorSequenceModel()
        res._vectors = mp.mpMaxVectorSequences(self._vectors, vs._vectors)
        return res

    def extractSequenceList(self):
        return mp.mpTransposeMatrix(self._vectors)

    def __str__(self):
        return '[\n'+'\n'.join([mp.mpVector(v) for v in self._vectors])+'\n]'

class MaxPlusMatrixModel(object):

    def __init__(self, rows=None):
        self._labels = []
        if rows is None:
            self._rows = []
        else:
            self._rows = rows

    def setLabels(self, labels):
        self._labels = labels

    def labels(self):
        return self._labels

    @staticmethod
    def fromMatrix(M):
        res = MaxPlusMatrixModel()
        res._rows = M
        return res

    def validate(self):
        pass

    def setMatrix(self, rows):
        for r in rows:
            self.addRow(r)

    def addRow(self, row):
        self._rows.append(row)

    def numberOfRows(self):
        return len(self._rows)

    def rows(self):
        return self._rows

    def numberOfColumns(self):
        if len(self._rows) == 0:
            return 0
        return len(self._rows[0])

    def isSquare(self):
        return self.numberOfRows() == self.numberOfColumns()

    def mpMatrix(self):
        '''
        Return the matrix (array of rows) in this MaxPlusMatrixModel
        '''
        return self._rows

    def __str__(self):
        return "[" + '\n'.join(['['+(', '.join([str(e) for e in r]))+']' for r in self._rows]) + ']'

    def eigenvectors(self):
        return mp.mpEigenVectors(self._rows)

    def eigenvalue(self):
        return mp.mpEigenValue(self._rows)

    def _precedencegraphLabels(self):
        return self.labels() if len(self.labels()) == self.numberOfRows() else [ 'x{}'.format(k) for k in range(self.numberOfRows())]

    def precedencegraph(self):
        return mp.mpPrecedenceGraph(self._rows, self._precedencegraphLabels())
            
    def precedencegraphGraphviz(self):
        return mp.mpPrecedenceGraphGraphviz(self._rows, self._precedencegraphLabels())

    def starClosure(self):
        try:
            cl = starClosure(self._rows)
        except PositiveCycleException:
            return False, None
        return True, cl

    def multiply(self, matOrVs):
        if isinstance(matOrVs, VectorSequenceModel):
            return VectorSequenceModel(mp.mpMultiplyMatrixVectorSequence(self._rows, matOrVs._vectors))
        else:
            return MaxPlusMatrixModel(mp.mpMultiplyMatrices(self._rows, matOrVs._rows))
        
    def vectortraceClosed(self, x0, ni):
        '''
        Compute a vector trace of state vectors of the matrix in this model. Requires that the matrix is square.

        Parameters
        ----------

        `x0`: the initial state, or `None` to use the zero vector as initial state

        `ni`: integer indicating the number of iterations to compute.

        '''
        matrices = {}
        matrices['A'] = self
        matrices['B'] = MaxPlusMatrixModel()
        matrices['C'] = MaxPlusMatrixModel()
        matrices['D'] = MaxPlusMatrixModel()
        return MaxPlusMatrixModel.vectortrace(matrices, x0, ni, [])

    @staticmethod
    def vectortrace(matrices, x0, ni, inputs, useEndingState = False):
        '''
        Compute a vector trace of the combined state vector and output vector, stacked

        Parameters
        ----------
        `matrices` : dictionary that maps the strings 'A', 'B', 'C' and 'D' to four MaxPlusMatrixModels representing state space matrices
        
        `x0`: the initial state vector, or `None` to default to the zero vector as the initial state
        
        `ni`: the number of iterations to compute, or `None` to compute the maximum number of iterations that can be computed given the input data

        `inputs`: a list of input event sequences

        `useEndingState`: boolean that determines if the state vector before, or after the iteration is used.

        '''
        
        def _validate(mat):
            if not ('A' in mat and 'B' in mat and 'C' in mat and 'D' in mat):
                raise Exception('Expected matrices A, B, C and D')
            A = mat['A']
            B = mat['B']
            C = mat['C']
            D = mat['D']
            if A.numberOfRows() != B.numberOfRows():
                if A.numberOfColumns() > 0 and B.numberOfColumns() > 0:
                    raise Exception('The number of rows of A does not match the number of rows of B')
            if C.numberOfRows() != D.numberOfRows():
                if C.numberOfColumns() > 0 and D.numberOfColumns() > 0:
                    raise Exception('The number of rows of C does not match the number of rows of D')
            if A.numberOfColumns() != C.numberOfColumns():
                if A.numberOfRows() > 0 and C.numberOfRows() > 0:
                    raise Exception('The number of columns of A does not match the number of columns of C')
            if B.numberOfColumns() != D.numberOfColumns():
                if B.numberOfRows() > 0 and D.numberOfRows() > 0:
                    raise Exception('The number of columns of B does not match the number of columns of D')
            return True

        if not _validate(matrices):
            raise Exception("Provided matrices are not a valid state-space model.")

        MA = matrices['A'].mpMatrix()
        MB = matrices['B'].mpMatrix()
        MC = matrices['C'].mpMatrix()
        MD = matrices['D'].mpMatrix()

        if len(inputs) < mp.mpNumberOfColumns(MB):
            raise Exception('Insufficient inputs sequences provided. ')
        if ni is None:
            ni = min([len(l) for l in inputs])
        for i in inputs:
            if len(i) < ni:
                raise Exception('Insufficiently long input sequences for {} iterations.'.format(ni))
        inpPref = [i[0:ni] for i in inputs]
        inputvectors = list(map(list, zip(*inpPref)))
        if len(inputvectors) ==0:
            # degenerate case of no inputs
            inputvectors = [[]] * ni

        if x0 is None:
            x = mp.mpZeroVector(mp.mpNumberOfRows(MA))
        else:
            x = x0

        trace = []
        for n in range(ni):
            nextX = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(MA, x), mp.mpMultiplyMatrixVector(MB, inputvectors[n]))
            outputvector = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(MC, x), mp.mpMultiplyMatrixVector(MD, inputvectors[n]))
            if useEndingState:
                stateVector = nextX
            else:
                stateVector = x
            trace.append(mp.mpStackVectors(inputvectors[n], mp.mpStackVectors(stateVector, outputvector)))
            x = nextX
        return trace

    @staticmethod
    def extractSequences(nSeq, uSeq, eventsequences, vectorsequences, inputLabels):
        '''
        Extract the input sequence for the input labels from the combined information from named and unnamed sequenced provided on the command line and the sequences defined in the model. Priority is given to sequences specified on the command line. Sequences that cannot be fined by name are filled with the unnamed sequences.
        '''

        if uSeq is None:
            uSeq = []
        if nSeq is None:
            allNamed = {}
        else:
            allNamed = nSeq.copy()

        # add event sequences
        for e in eventsequences:
            allNamed[e] = eventsequences[e].sequence()

        # add vector sequences.
        for v in vectorsequences:
            vsl = vectorsequences[v].extractSequenceList()
            for k in range(len(vsl)):
                allNamed[v+str(k+1)] = vsl[k]

        res = []
        uInd = 0
        for l in inputLabels:
            if l in allNamed:
                res.append(allNamed[l])
            else:
                if uInd >= len(uSeq):
                    raise Exception("Insufficient input sequences specified")
                res.append(uSeq[uInd])
                uInd += 1
        return res

    @staticmethod
    def multiplySequence(matrices):
        return reduce(lambda prod, mat: prod.multiply(mat), matrices)
        

    @staticmethod
    def fromDSL(dslString):

        factory = dict()
        factory['Init'] = lambda : MaxPlusMatrixModel()
        factory['AddRow'] = lambda mpm, r: mpm.addRow(r)
        factory['InitVectorSequence'] = lambda : VectorSequenceModel()
        factory['InitEventSequence'] = lambda : EventSequenceModel()
        factory['AddLabels'] = lambda m, labels: m.setLabels(labels)

        name, resMatrices, resVectorSequences, resEventSequences = parseMPMDSL(dslString, factory)
        return name, resMatrices, resVectorSequences, resEventSequences
        


    def asDSL(self, name, allInstances = None):

        # create string writer for the output
        output = StringIO()
        if allInstances is None:
            output.write("max-plus model {} : \nmatrices\nA = [\n".format(name))
            for r in self._rows:
                output.write("\t{}\n".format(mp.mpVector(r)))
            output.write("]\n")
        else:
            mats = [i for i in allInstances if isinstance(allInstances[i], MaxPlusMatrixModel)]
            vSequences = [i for i in allInstances if isinstance(allInstances[i], VectorSequenceModel)]
            eSequences = [i for i in allInstances if isinstance(allInstances[i], EventSequenceModel)]

            output.write("max-plus model {} : \n".format(name))

            if len(mats) > 0:
                output.write("\nmatrices\n")
                for mat in mats:
                    if len(allInstances[mat].labels()) > 0:
                        labels = '({})'.format(' '.join(allInstances[mat].labels()))
                    else:
                        labels=''
                    output.write("{} {} = [\n".format(mat, labels))
                    for r in allInstances[mat].rows():
                        output.write("\t{}\n".format(mp.mpVector(r)))
                    output.write("]\n")


            if len(vSequences) > 0:
                output.write("\nvector sequences\n")
                for vs in vSequences:
                    if len(allInstances[vs].labels()) > 0:
                        labels = '({})'.format(' '.join(allInstances[vs].labels()))
                    else:
                        labels=''
                    output.write("{} {} = [\n".format(vs, labels))
                    for v in allInstances[vs].vectors():
                        output.write("\t{}\n".format(mp.mpVector(v)))
                    output.write("]\n")

            if len(eSequences) > 0:
                output.write("\nevent sequences\n")
                for es in eSequences:
                    output.write("{} = {}".format(es, mp.mpVector(allInstances[es].sequence())))
        
        result = output.getvalue()
        output.close()
        return result

