from fractions import Fraction
from functools import reduce
from io import StringIO
from typing import List, Literal, Optional, Tuple, Union, Dict
import dataflow.maxplus.maxplus as mp
from dataflow.maxplus.utils.printing import mpVectorToString
from dataflow.libmpmgrammar import parseMPMDSL
from dataflow.maxplus.starclosure import PositiveCycleException
import pygraph.classes.digraph  as pyg

# Represent a finite event sequence, i.e., a list of time stamps
class MPMException(mp.MPException):
    pass

class MPMValidateException(MPMException):
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
            raise MPMException('Object of the wrong type to take convolution with event sequence.')
        res = EventSequenceModel()
        res._sequence = mp.mpConvolution(self._sequence, es._sequence)
        return res

    def maxWith(self, es: 'EventSequenceModel') -> 'EventSequenceModel':
        '''Compute the maximum with another event sequence'''
        if not isinstance(es, EventSequenceModel):
            raise MPMException('Object of the wrong type to take maximum with event sequence.')
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
        return mpVectorToString(self._sequence)

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

    def __len__(self) -> int:
        return self.length()

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

    def multiply(self, m: Union['VectorSequenceModel','MaxPlusMatrixModel']):
        raise MPMException("Cannot multiply vector sequence on the left-hand side.")

    def __str__(self)->str:
        return '[\n'+'\n'.join([mpVectorToString(v) for v in self._vectors])+'\n]'

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

    def eigenvectors(self) -> Tuple[List[Tuple[mp.TMPVector,Fraction]],List[Tuple[mp.TMPVector,mp.TMPVector]]]:
        '''
        Compute eigenvectors and generalized eigenvector and corresponding (generalized) eigenvalues.
        Returns a pair with:
        A list of pairs of eigenvector and eigenvalue and a list of pairs of generalized eigenvectors and generalized eigenvalues.
        '''
        if not self.isSquare():
            raise MPMException("Matrix must be square to compute eigenvectors.")
        return mp.mpEigenVectors(self._rows)

    def eigenvalue(self) -> mp.TTimeStamp:
        '''Determine the largest eigenvalue of the matrix.'''
        return mp.mpEigenValue(self._rows)

    def _precedenceGraphLabels(self) -> List[str]:
        '''Return a lst of labels for the rows/columns of the matrix. If none have been explicitly defined it generate labels `x` with a number.'''
        return self.labels() if len(self.labels()) == self.numberOfRows() else [ 'x{}'.format(k) for k in range(self.numberOfRows())]

    def precedenceGraph(self) -> pyg.digraph:
        '''Determine the precedence graph of the matrix.'''
        return mp.mpPrecedenceGraph(self._rows, self._precedenceGraphLabels())
            
    def precedenceGraphGraphviz(self) -> str:
        '''Return a Graphviz representation of the precedence graph as a string.'''
        return mp.mpPrecedenceGraphGraphviz(self._rows, self._precedenceGraphLabels())

    def starClosure(self)->Union[Tuple[Literal[False],None],Tuple[Literal[True],'MaxPlusMatrixModel']]:
        '''Determine the * closure. If it exist return True and the star close. If it doesn't return False and None.'''
        try:
            cl = MaxPlusMatrixModel(mp.mpStarClosure(self._rows))
        except PositiveCycleException:
            return False, None
        return True, cl

    def multiply(self, matOrVs: Union['MaxPlusMatrixModel',VectorSequenceModel]) -> Union['MaxPlusMatrixModel',VectorSequenceModel]:
        '''Multiply the matrix with another matrix or with a vector sequence.'''
        if isinstance(matOrVs, VectorSequenceModel):
            return VectorSequenceModel(mp.mpMultiplyMatrixVectorSequence(self._rows, matOrVs._vectors))
        else:
            return MaxPlusMatrixModel(mp.mpMultiplyMatrices(self._rows, matOrVs._rows))
        
    def vectorTraceClosed(self, x0: mp.TMPVector, ni: int) -> mp.TMPVectorList:
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
    def vectorTrace(matrices: Dict[str,'MaxPlusMatrixModel'], x0: mp.TMPVector, ni: int, inputs: List[mp.TTimeStampList], useEndingState: bool = False) -> mp.TMPVectorList:
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
        
        def _validate(mat: Dict[str,'MaxPlusMatrixModel'])->None:
            if not ('A' in mat and 'B' in mat and 'C' in mat and 'D' in mat):
                raise MPMValidateException('Expected matrices A, B, C and D')
            A = mat['A']
            B = mat['B']
            C = mat['C']
            D = mat['D']
            if A.numberOfRows() != B.numberOfRows():
                if A.numberOfColumns() > 0 and B.numberOfColumns() > 0:
                    raise MPMValidateException('The number of rows of A does not match the number of rows of B')
            if C.numberOfRows() != D.numberOfRows():
                if C.numberOfColumns() > 0 and D.numberOfColumns() > 0:
                    raise MPMValidateException('The number of rows of C does not match the number of rows of D')
            if A.numberOfColumns() != C.numberOfColumns():
                if A.numberOfRows() > 0 and C.numberOfRows() > 0:
                    raise MPMValidateException('The number of columns of A does not match the number of columns of C')
            if B.numberOfColumns() != D.numberOfColumns():
                if B.numberOfRows() > 0 and D.numberOfRows() > 0:
                    raise MPMValidateException('The number of columns of B does not match the number of columns of D')

        try:
            _validate(matrices)
        except MPMValidateException:
            raise MPMException("Provided matrices are not a valid state-space model.")

        MA = matrices['A'].mpMatrix()
        MB = matrices['B'].mpMatrix()
        MC = matrices['C'].mpMatrix()
        MD = matrices['D'].mpMatrix()

        if len(inputs) < mp.mpNumberOfColumns(MB):
            raise MPMException('Insufficient inputs sequences provided. ')
        if ni is None:
            ni = min([len(l) for l in inputs])
        for i in inputs:
            if len(i) < ni:
                raise MPMException('Insufficiently long input sequences for {} iterations.'.format(ni))
        inpPref = [i[0:ni] for i in inputs]
        inputVectors: mp.TMPVectorList = list(map(list, zip(*inpPref)))
        if len(inputVectors) ==0:
            # degenerate case of no inputs
            inputVectors = [[]] * ni

        x: mp.TMPVector
        if x0 is None:
            x = mp.mpZeroVector(mp.mpNumberOfRows(MA))
        else:
            x = x0

        trace: mp.TMPVectorList = []
        for n in range(ni):
            nextX: mp.TMPVector = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(MA, x), mp.mpMultiplyMatrixVector(MB, inputVectors[n]))
            outputVector = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(MC, x), mp.mpMultiplyMatrixVector(MD, inputVectors[n]))
            if useEndingState:
                stateVector = nextX
            else:
                stateVector = x
            trace.append(mp.mpStackVectors(inputVectors[n], mp.mpStackVectors(stateVector, outputVector)))
            x = nextX
        return trace

    @staticmethod
    def extractSequences(nSeq: Optional[Dict[str,mp.TTimeStampList]], uSeq: Optional[List[mp.TTimeStampList]], eventSequences: Dict[str,EventSequenceModel], vectorSequences: Dict[str,VectorSequenceModel], inputLabels: List[str]) -> List[mp.TTimeStampList]:
        '''
        Extract the input sequence for the input labels from the combined information from named (nSeq) and unnamed (uSeq) sequences provided on the command line and the sequences defined in the model. Priority is given to sequences specified on the command line. Sequences that cannot be found by name are filled with the unnamed sequences.
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

        res: List[mp.TTimeStampList] = []
        uInd = 0
        for l in inputLabels:
            if l in allNamed:
                res.append(allNamed[l])
            else:
                if uInd >= len(uSeq):
                    raise MPMException("Insufficient input sequences specified")
                res.append(uSeq[uInd])
                uInd += 1
        return res

    @staticmethod
    def multiplySequence(matrices: List[Union['MaxPlusMatrixModel',VectorSequenceModel]])->Union['MaxPlusMatrixModel',VectorSequenceModel]:
        '''Multiply the sequence of matrices, possibly ending with a vector sequence.'''
        return reduce(lambda prod, mat: prod.multiply(mat), matrices)
        
    @staticmethod
    def fromDSL(dslString: str)-> Tuple[str,Dict[str,'MaxPlusMatrixModel'],Dict[str,VectorSequenceModel],Dict[str,EventSequenceModel]]:
        '''
        Parse dslString and extract model name, matrices, vector sequences and event sequences.
        Raise an exception if the parsing fails. 
        '''

        factory = dict()
        factory['Init'] = lambda : MaxPlusMatrixModel()
        factory['AddRow'] = lambda mpm, r: mpm.addRow(r)
        factory['InitVectorSequence'] = lambda : VectorSequenceModel()
        factory['InitEventSequence'] = lambda : EventSequenceModel()
        factory['AddLabels'] = lambda m, labels: m.setLabels(labels)

        name, resMatrices, resVectorSequences, resEventSequences = parseMPMDSL(dslString, factory)
        if name is None or resMatrices is None or resVectorSequences is None or resEventSequences is None:
            raise MPMException("Failed to parse max-plus model")
        return name, resMatrices, resVectorSequences, resEventSequences


    def asDSL(self, name: str, allInstances: Optional[Dict[str,Union['MaxPlusMatrixModel',VectorSequenceModel,EventSequenceModel]]] = None) -> str:

        '''
        Convert the receiver and all entities in optional allInstances to a string in the MPM Domain Specific Language using the provided name as the name for the model.
        '''

        # create string writer for the output
        output = StringIO()
        if allInstances is None:
            output.write("max-plus model {} : \nmatrices\nA = [\n".format(name))
            for r in self._rows:
                output.write("\t{}\n".format(mpVectorToString(r)))
            output.write("]\n")
        else:
            mats: List[str] = [i for i in allInstances if isinstance(allInstances[i], MaxPlusMatrixModel)]
            vSequences = [i for i in allInstances if isinstance(allInstances[i], VectorSequenceModel)]
            eSequences = [i for i in allInstances if isinstance(allInstances[i], EventSequenceModel)]

            output.write("max-plus model {} : \n".format(name))

            if len(mats) > 0:
                output.write("\nmatrices\n")
                for mat in mats:
                    mm: MaxPlusMatrixModel = allInstances[mat]  # type: ignore
                    if len(mm.labels()) > 0:
                        labels = '({})'.format(' '.join(mm.labels()))
                    else:
                        labels = ''
                    output.write("{} {} = [\n".format(mat, labels))
                    for r in mm.rows():
                        output.write("\t{}\n".format(mpVectorToString(r)))
                    output.write("]\n")

            if len(vSequences) > 0:
                output.write("\nvector sequences\n")
                for vs in vSequences:
                    vsm: VectorSequenceModel = allInstances[vs]  # type: ignore
                    if len(vsm.labels()) > 0:
                        labels = '({})'.format(' '.join(vsm.labels()))
                    else:
                        labels=''
                    output.write("{} {} = [\n".format(vs, labels))
                    for v in vsm.vectors():
                        output.write("\t{}\n".format(mpVectorToString(v)))
                    output.write("]\n")

            if len(eSequences) > 0:
                output.write("\nevent sequences\n")
                for es in eSequences:
                    esm: EventSequenceModel = allInstances[es]  # type: ignore
                    output.write("{} = {}".format(es, mpVectorToString(esm.sequence())))
        
        result = output.getvalue()
        output.close()
        return result

