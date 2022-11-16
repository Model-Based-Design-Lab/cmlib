from functools import reduce
from io import StringIO
from typing import List, Optional, Tuple
import dataflow.maxplus.maxplus as mp
from dataflow.libmpmgrammar import parseMPMDSL
from dataflow.maxplus.starclosure import PositiveCycleException, starClosure

# Represent a finite event sequence, i.e., a list of time stamps

class MPMValidateException(mp.MPException):
    pass
class EventSequenceModel(object):

    _sequence: mp.TTimeStampList

    def __init__(self, seq: mp.TTimeStampList = []):
        # _sequence captures the event sequence
        self._sequence = seq
    
    def validate(self):
        # no checks needed
        pass

    def addRow(self, row: mp.TTimeStampList):
        ''' bit of a hack for the parser to reuse code. Do not use. '''
        # TODO: get rid of this method by making the parse call an appropriate function
        self._sequence = row

    def sequence(self):
        '''Get the time stamp sequence'''
        return self._sequence

    def length(self)->int:
        '''Returns the length of the sequence'''
        return len(self._sequence)

    def convolveWith(self: 'EventSequenceModel', es: 'EventSequenceModel')->'EventSequenceModel':
        '''Compute the convolution with another EventSequenceModel'''
        if not isinstance(es, EventSequenceModel):
            raise Exception('Object of the wrong type to take convolution with event sequence.')
        res = EventSequenceModel()
        res._sequence = mp.mpConvolution(self._sequence, es._sequence)
        return res

    def maxWith(self, es: 'EventSequenceModel') -> 'EventSequenceModel':
        '''Compute the maximum with another event sequence'''
        if not isinstance(es, EventSequenceModel):
            raise Exception('Object of the wrong type to take maximum with event sequence.')
        res = EventSequenceModel()
        res._sequence = mp.mpMaxEventSequences(self._sequence, es._sequence)
        return res

    def delay(self, n: int)->'EventSequenceModel':
        '''Compute the delayed event sequence, delayed by n samples'''
        res = EventSequenceModel()
        res._sequence = mp.mpDelay(self._sequence, n)
        return res

    def scale(self, c: mp.TTimeStamp)->'EventSequenceModel':
        '''Compute a scaled version of the event sequence, i.e., add constant to every element.'''
        res = EventSequenceModel()
        res._sequence = mp.mpScale(self._sequence, c)
        return res

    def extractSequenceList(self)->List[mp.TTimeStampList]:
        '''Determine list of sequences. For an event sequence, the list will contain only one list as an element'''
        return [self._sequence]

    def __str__(self)->str:
        '''Return a string representation.'''
        return mp.mpVector(self._sequence)

class VectorSequenceModel(object):

    _vectors: mp.TMPVectorList
    _labels: List[str]

    def __init__(self, vectors: Optional[mp.TMPVectorList] = None):
        if vectors is None:
            self._vectors: mp.TMPVectorList = []
        else:
            self._vectors = vectors
        self._labels: List[str] = []

    def addVector(self, v: mp.TMPVector):
        '''Add a vector to the vector sequence'''
        self._vectors.append(v)

    def addRow(self, row: mp.TMPVector):
        '''Cheat method to reuse matrix like methods. Do not use.'''
        # TODO: remove method divert to calls to addVector
        self._vectors.append(row)

    def setLabels(self, labels: List[str]):
        '''Set labels for the vector elements. The length of the list of labels should match the size of the vectors in the list'''
        self._labels = labels

    def labels(self)->List[str]:
        '''Get the labels of the vector elements'''
        return self._labels

    def getLabel(self, n: int, base: str = 's'):
        '''Get the label of element n (starting from 0) of the vectors. If the label has not been previously set, a default is constructed from the base with a number.'''
        if n < len(self._labels):
            return self._labels[n]
        else:
            return base + str(n+1)

    def vectors(self) -> mp.TMPVectorList:
        '''Get the list of vectors.'''
        return self._vectors

    def length(self) -> int:
        '''Returns the number of vector in the list.'''
        return len(self._vectors)

    def vectorLength(self) -> int:
        '''Returns the length of the vectors in the list. Assumes that all vectors in the list have the same length. Returns 0 if there are no vectors in the list.'''
        if len(self._vectors) == 0:
            return 0
        else:
            return len(self._vectors[0])

    def validate(self):
        '''Check if all vectors are of the same length. Raise an MPMValidateException if the vector sequence does not validate.'''
        if len(set([len(r) for r in self._vectors])) > 1:
            raise MPMValidateException("Vector sequence contains vectors of unequal length.")

    def maxWith(self, vs: 'VectorSequenceModel')->'VectorSequenceModel':
        '''Compute the maximum with another vector sequence.'''
        res = VectorSequenceModel()
        res._vectors = mp.mpMaxVectorSequences(self._vectors, vs._vectors)
        return res

    def extractSequenceList(self)->List[mp.TTimeStampList]:
        '''Convert vector sequence into a list of event sequences from the corresponding elements from each of the vectors.'''
        return mp.mpTransposeMatrix(self._vectors)

    def __str__(self)->str:
        return '[\n'+'\n'.join([mp.mpVector(v) for v in self._vectors])+'\n]'

class MaxPlusMatrixModel(object):
    '''Model of a max-plus matrix'''

    _labels: List[str]
    _rows: List[mp.TMPVector]

    def __init__(self, rows: Optional[List[mp.TMPVector]]=None):
        self._labels = []
        if rows is None:
            self._rows = []
        else:
            self._rows = rows

    def setLabels(self, labels: List[str]):
        '''Set labels for the rows / columns of the matrix'''
        self._labels = labels

    def labels(self) -> List[str]:
        '''Get the row/column labels'''
        return self._labels

    @staticmethod
    def fromMatrix(M: mp.TMPMatrix)->'MaxPlusMatrixModel':
        '''Create a matrix model from a matrix in the form of a list of row vectors'''
        res = MaxPlusMatrixModel()
        res._rows = M
        return res

    def validate(self):
        '''Validate any constraints on the matrix.'''
        if len(self._rows) == 0:
            return
        n = self.numberOfColumns()
        for r in self._rows:
            if len(r) != n:
                raise MPMValidateException("Matrix contains rows of unequal length.")

    def setMatrix(self, rows: mp.TMPMatrix):
        '''Add rows to the matrix'''
        for r in rows:
            self.addRow(r)

    def addRow(self, row: mp.TMPVector):
        '''Add a row vector to the matrix.'''
        self._rows.append(row)

    def numberOfRows(self) -> int:
        '''Returns the number of row of the matrix.'''
        return len(self._rows)

    def rows(self) -> mp.TMPMatrix:
        '''Return the matrix as a list of row vectors.'''
        return self._rows

    def numberOfColumns(self) -> int:
        '''Return the number of columns of the matrix'''
        if len(self._rows) == 0:
            return 0
        return len(self._rows[0])

    def isSquare(self) -> bool:
        '''Check if the matrix is square.'''
        return self.numberOfRows() == self.numberOfColumns()

    def mpMatrix(self) -> mp.TMPMatrix:
        '''
        Return the matrix (list of rows) in this MaxPlusMatrixModel
        '''
        return self.rows()

    def __str__(self) -> str:
        return "[" + '\n'.join(['['+(', '.join([str(e) for e in r]))+']' for r in self._rows]) + ']'

    def eigenvectors(self) -> Tuple[List[Tuple[mp.TMPVector,float]],List[Tuple[mp.TMPVector,mp.TMPVector]]]:
        '''
        Compute eigenvectors and generalized eigenvector and corresponding (generalized) eigenvalues.
        Returns a pair with:
        A list of pairs of eigenvector and eigenvalue and a list of pairs of generalized eigenvectors and generalized eigenvalues.
        '''
        if not self.isSquare():
            raise mp.MPException("Matrix must be square to compute eigenvectors.")
        return mp.mpEigenVectors(self._rows)

    def eigenvalue(self):
        return mp.mpEigenValue(self._rows)

    def _precedenceGraphLabels(self):
        return self.labels() if len(self.labels()) == self.numberOfRows() else [ 'x{}'.format(k) for k in range(self.numberOfRows())]

    def precedenceGraph(self):
        return mp.mpPrecedenceGraph(self._rows, self._precedenceGraphLabels())
            
    def precedenceGraphGraphviz(self):
        return mp.mpPrecedenceGraphGraphviz(self._rows, self._precedenceGraphLabels())

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
        
    def vectorTraceClosed(self, x0, ni):
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
        return MaxPlusMatrixModel.vectorTrace(matrices, x0, ni, [])

    @staticmethod
    def vectorTrace(matrices, x0, ni, inputs, useEndingState = False):
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
        inputVectors = list(map(list, zip(*inpPref)))
        if len(inputVectors) ==0:
            # degenerate case of no inputs
            inputVectors = [[]] * ni

        if x0 is None:
            x = mp.mpZeroVector(mp.mpNumberOfRows(MA))
        else:
            x = x0

        trace = []
        for n in range(ni):
            nextX = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(MA, x), mp.mpMultiplyMatrixVector(MB, inputVectors[n]))
            outputVector = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(MC, x), mp.mpMultiplyMatrixVector(MD, inputVectors[n]))
            if useEndingState:
                stateVector = nextX
            else:
                stateVector = x
            trace.append(mp.mpStackVectors(inputVectors[n], mp.mpStackVectors(stateVector, outputVector)))
            x = nextX
        return trace

    @staticmethod
    def extractSequences(nSeq, uSeq, eventSequences, vectorSequences, inputLabels):
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
        for e in eventSequences:
            allNamed[e] = eventSequences[e].sequence()

        # add vector sequences.
        for v in vectorSequences:
            vsl = vectorSequences[v].extractSequenceList()
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

