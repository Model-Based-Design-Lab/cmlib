"""Library for max-plus algebra models"""

from fractions import Fraction
from functools import reduce
from io import StringIO
from typing import List, Literal, Optional, Tuple, Union, Dict
import dataflow.maxplus.maxplus as mp
from dataflow.maxplus.utils.printing import mpPrettyVectorToString, \
      mpVectorToFractionString, commonDenominatorMatrix, isComplex, mpVectorToString
from dataflow.libmpmgrammar import parseMPMDSL
from dataflow.maxplus.starclosure import PositiveCycleException
import pygraph.classes.digraph  as pyg

class MPMException(mp.MPException):
    """Exceptions in this library."""

class MPMValidateException(MPMException):
    """Validation exception"""

class EventSequenceModel:
    """Model of an event sequence."""

    _sequence: mp.TTimeStampList

    def __init__(self, seq: Union[mp.TTimeStampList,None] = None):
        # _sequence captures the event sequence
        if seq is None:
            seq = []
        self._sequence = seq

    def validate(self):
        """Validate the model."""
        # no checks needed

    def set_sequence(self, seq: mp.TTimeStampList):
        ''' Set the event sequence '''
        self._sequence = seq

    def sequence(self):
        '''Get the time stamp sequence'''
        return self._sequence

    def length(self)->int:
        '''Returns the length of the sequence'''
        return len(self._sequence)

    def convolve_with(self: 'EventSequenceModel', es: 'EventSequenceModel')->'EventSequenceModel':
        '''Compute the convolution with another EventSequenceModel'''
        if not isinstance(es, EventSequenceModel):
            raise MPMException('Object of the wrong type to take convolution with event sequence.')
        res = EventSequenceModel()
        res.set_sequence(mp.mpConvolution(self._sequence, es.sequence()))
        return res

    def max_with(self, es: 'EventSequenceModel') -> 'EventSequenceModel':
        '''Compute the maximum with another event sequence'''
        if not isinstance(es, EventSequenceModel):
            raise MPMException('Object of the wrong type to take maximum with event sequence.')
        res = EventSequenceModel()
        res.set_sequence(mp.mpMaxEventSequences(self._sequence, es.sequence()))
        return res

    def delay(self, n: int)->'EventSequenceModel':
        '''Compute the delayed event sequence, delayed by n samples'''
        res = EventSequenceModel()
        res.set_sequence(mp.mpDelay(self._sequence, n))
        return res

    def scale(self, c: mp.TTimeStamp)->'EventSequenceModel':
        '''Compute a scaled version of the event sequence, i.e., add constant to every element.'''
        res = EventSequenceModel()
        res.set_sequence(mp.mpScale(self._sequence, c))
        return res

    def extract_sequence_list(self)->List[mp.TTimeStampList]:
        '''Determine list of sequences. For an event sequence, the list will
        contain only one list as an element'''
        return [self._sequence]

    def __str__(self)->str:
        '''Return a string representation.'''
        return mpPrettyVectorToString(self._sequence)

class VectorSequenceModel:
    """Model of a sequence of vectors."""

    _vectors: mp.TMPVectorList
    _labels: List[str]

    def __init__(self, vectors: Optional[mp.TMPVectorList] = None):
        if vectors is None:
            self._vectors: mp.TMPVectorList = []
        else:
            self._vectors = vectors
        self._labels: List[str] = []

    def add_vector(self, v: mp.TMPVector):
        '''Add a vector to the vector sequence'''
        self._vectors.append(v)

    def set_labels(self, labels: List[str]):
        '''Set labels for the vector elements. The length of the list of labels
        should match the size of the vectors in the list'''
        self._labels = labels

    def labels(self)->List[str]:
        '''Get the labels of the vector elements'''
        return self._labels

    def get_label(self, n: int, base: str = 's'):
        '''Get the label of element n (starting from 0) of the vectors. If the label has
        not been previously set, a default is constructed from the base with a number.'''
        if n < len(self._labels):
            return self._labels[n]
        return base + str(n+1)

    def vectors(self) -> mp.TMPVectorList:
        '''Get the list of vectors.'''
        return self._vectors

    def set_vectors(self, v: mp.TMPVectorList):
        '''Set the list of vectors.'''
        self._vectors = v

    def length(self) -> int:
        '''Returns the number of vector in the list.'''
        return len(self._vectors)

    def __len__(self) -> int:
        return self.length()

    def vector_length(self) -> int:
        '''Returns the length of the vectors in the list. Assumes that all vectors in
        the list have the same length. Returns 0 if there are no vectors in the list.'''
        if len(self._vectors) == 0:
            return 0
        return len(self._vectors[0])

    def validate(self):
        '''Check if all vectors are of the same length. Raise an MPMValidateException
        if the vector sequence does not validate.'''
        if len({len(r) for r in self._vectors}) > 1:
            raise MPMValidateException("Vector sequence contains vectors of unequal length.")

    def max_with(self, vs: 'VectorSequenceModel')->'VectorSequenceModel':
        '''Compute the maximum with another vector sequence.'''
        res = VectorSequenceModel()
        res.set_vectors(mp.mpMaxVectorSequences(self._vectors, vs.vectors()))
        return res

    def extract_sequence_list(self)->List[mp.TTimeStampList]:
        '''Convert vector sequence into a list of event sequences from the
        corresponding elements from each of the vectors.'''
        return mp.mpTransposeMatrix(self._vectors)

    def multiply(self, m: Union['VectorSequenceModel','MaxPlusMatrixModel']):
        """Multiply with a model of vector sequence or a matrix."""
        raise MPMException("Cannot multiply vector sequence on the left-hand side.")

    def __str__(self)->str:
        return '[\n'+'\n'.join([mpPrettyVectorToString(v) for v in self._vectors])+'\n]'

class MaxPlusMatrixModel:
    '''Model of a max-plus matrix'''

    _labels: List[str]
    _rows: List[mp.TMPVector]

    def __init__(self, rows: Optional[List[mp.TMPVector]]=None):
        self._labels = []
        if rows is None:
            self._rows = []
        else:
            self._rows = rows

    def set_labels(self, labels: List[str]):
        '''Set labels for the rows / columns of the matrix'''
        self._labels = labels

    def labels(self) -> List[str]:
        '''Get the row/column labels'''
        return self._labels

    def set_rows(self, rows: List[mp.TMPVector]):
        '''Set rows of the matrix'''
        self._rows = rows

    def rows(self) -> List[mp.TMPVector]:
        '''Get the rows'''
        return self._rows

    @staticmethod
    def from_matrix(matrix: mp.TMPMatrix)->'MaxPlusMatrixModel':
        '''Create a matrix model from a matrix in the form of a list of row vectors'''
        res = MaxPlusMatrixModel()
        res.set_rows(matrix)
        return res

    def validate(self):
        '''Validate any constraints on the matrix.'''
        if len(self._rows) == 0:
            return
        n = self.number_of_columns()
        for r in self._rows:
            if len(r) != n:
                raise MPMValidateException("Matrix contains rows of unequal length.")

    def set_matrix(self, rows: mp.TMPMatrix):
        '''Add rows to the matrix'''
        for r in rows:
            self.add_row(r)

    def add_row(self, row: mp.TMPVector):
        '''Add a row vector to the matrix.'''
        self._rows.append(row)

    def number_of_rows(self) -> int:
        '''Returns the number of row of the matrix.'''
        return len(self._rows)

    def number_of_columns(self) -> int:
        '''Return the number of columns of the matrix'''
        if len(self._rows) == 0:
            return 0
        return len(self._rows[0])

    def is_square(self) -> bool:
        '''Check if the matrix is square.'''
        return self.number_of_rows() == self.number_of_columns()

    def mp_matrix(self) -> mp.TMPMatrix:
        '''
        Return the matrix (list of rows) in this MaxPlusMatrixModel
        '''
        return self.rows()

    def __str__(self) -> str:
        return "[" + '\n'.join(['['+(', '.join([str(e) for e in r]))+']' for r in self._rows]) + ']'

    def eigenvectors(self) -> Tuple[List[Tuple[mp.TMPVector,Fraction]], \
                                    List[Tuple[mp.TMPVector,mp.TMPVector]]]:
        '''
        Compute eigenvectors and generalized eigenvector and corresponding (generalized)
        eigenvalues.
        Returns a pair with:
        A list of pairs of eigenvector and eigenvalue and a list of pairs of generalized
        eigenvectors and generalized eigenvalues.
        '''
        if not self.is_square():
            raise MPMException("Matrix must be square to compute eigenvectors.")
        return mp.mpEigenVectors(self._rows)

    def eigenvalue(self) -> mp.TTimeStamp:
        '''Determine the largest eigenvalue of the matrix.'''
        return mp.mpEigenValue(self._rows)

    def _precedence_graph_labels(self) -> List[str]:
        '''Return a lst of labels for the rows/columns of the matrix. If none have been
        explicitly defined it generate labels `x` with a number.'''
        return self.labels() if len(self.labels()) == self.number_of_rows() else \
            [ f'x{k}' for k in range(self.number_of_rows())]

    def precedence_graph(self) -> pyg.digraph:
        '''Determine the precedence graph of the matrix.'''
        return mp.mpPrecedenceGraph(self._rows, self._precedence_graph_labels())

    def precedence_graph_graphviz(self) -> str:
        '''Return a Graphviz representation of the precedence graph as a string.'''
        return mp.mpPrecedenceGraphGraphviz(self._rows, self._precedence_graph_labels())

    def star_closure(self)->Union[Tuple[Literal[False],None],Tuple[Literal[True], \
                                                                   'MaxPlusMatrixModel']]:
        '''Determine the * closure. If it exist return True and the star close.
        If it doesn't return False and None.'''
        try:
            cl = MaxPlusMatrixModel(mp.mpStarClosure(self._rows))
        except PositiveCycleException:
            return False, None
        return True, cl

    def multiply(self, mat_or_vs: Union['MaxPlusMatrixModel',VectorSequenceModel]) -> \
        Union['MaxPlusMatrixModel',VectorSequenceModel]:
        '''Multiply the matrix with another matrix or with a vector sequence.'''
        if isinstance(mat_or_vs, VectorSequenceModel):
            return VectorSequenceModel(mp.mpMultiplyMatrixVectorSequence(self._rows, \
                                                                         mat_or_vs.vectors()))
        return MaxPlusMatrixModel(mp.mpMultiplyMatrices(self._rows, mat_or_vs.rows()))

    def vector_trace_closed(self, x0: mp.TMPVector, ni: int) -> mp.TMPVectorList:
        '''
        Compute a vector trace of state vectors of the matrix in this model. Requires
        that the matrix is square.

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
        return MaxPlusMatrixModel.vector_trace(matrices, x0, ni, [])

    @staticmethod
    def vector_trace(matrices: Dict[str,'MaxPlusMatrixModel'], x0: mp.TMPVector, ni: int, \
                     inputs: List[mp.TTimeStampList], use_ending_state: bool = False) -> \
                        mp.TMPVectorList:
        '''
        Compute a vector trace of the combined state vector and output vector, stacked

        Parameters
        ----------
        `matrices` : dictionary that maps the strings 'A', 'B', 'C' and 'D' to four
        MaxPlusMatrixModels representing state space matrices

        `x0`: the initial state vector, or `None` to default to the zero vector as the initial state

        `ni`: the number of iterations to compute, or `None` to compute the maximum number of
        iterations that can be computed given the input data

        `inputs`: a list of input event sequences

        `useEndingState`: boolean that determines if the state vector before, or after the
        iteration is used.

        '''

        def _validate(mat: Dict[str,'MaxPlusMatrixModel'])->None:
            if not ('A' in mat and 'B' in mat and 'C' in mat and 'D' in mat):
                raise MPMValidateException('Expected matrices A, B, C and D')
            matrix_s = mat['A']
            matrix_b = mat['B']
            matrix_c = mat['C']
            matrix_d = mat['D']
            if matrix_s.number_of_rows() != matrix_b.number_of_rows():
                if matrix_s.number_of_columns() > 0 and matrix_b.number_of_columns() > 0:
                    raise MPMValidateException('The number of rows of A does not match ' \
                                               'the number of rows of B')
            if matrix_c.number_of_rows() != matrix_d.number_of_rows():
                if matrix_c.number_of_columns() > 0 and matrix_d.number_of_columns() > 0:
                    raise MPMValidateException('The number of rows of C does not match ' \
                                               'the number of rows of D')
            if matrix_s.number_of_columns() != matrix_c.number_of_columns():
                if matrix_s.number_of_rows() > 0 and matrix_c.number_of_rows() > 0:
                    raise MPMValidateException('The number of columns of A does not match ' \
                                               'the number of columns of C')
            if matrix_b.number_of_columns() != matrix_d.number_of_columns():
                if matrix_b.number_of_rows() > 0 and matrix_d.number_of_rows() > 0:
                    raise MPMValidateException('The number of columns of B does not match ' \
                                               'the number of columns of D')

        try:
            _validate(matrices)
        except MPMValidateException:
            raise MPMException("Provided matrices are not a valid state-space model.") # pylint: disable=raise-missing-from

        matrix_a = matrices['A'].mp_matrix()
        matrix_b = matrices['B'].mp_matrix()
        matrix_c = matrices['C'].mp_matrix()
        matrix_d = matrices['D'].mp_matrix()

        if len(inputs) < mp.mpNumberOfColumns(matrix_b):
            raise MPMException('Insufficient inputs sequences provided. ')
        if ni is None:
            ni = min([len(l) for l in inputs])
        for i in inputs:
            if len(i) < ni:
                raise MPMException(f'Insufficiently long input sequences for {ni} iterations.')
        inp_pref = [i[0:ni] for i in inputs]
        input_vectors: mp.TMPVectorList = list(map(list, zip(*inp_pref)))
        if len(input_vectors) ==0:
            # degenerate case of no inputs
            input_vectors = [[]] * ni

        x: mp.TMPVector
        if x0 is None:
            x = mp.mpZeroVector(mp.mpNumberOfRows(matrix_a))
        else:
            x = x0

        trace: mp.TMPVectorList = []
        for n in range(ni):
            next_x: mp.TMPVector = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(matrix_a, x), \
                                        mp.mpMultiplyMatrixVector(matrix_b, input_vectors[n]))
            output_vector = mp.mpMaxVectors(mp.mpMultiplyMatrixVector(matrix_c, x), \
                                            mp.mpMultiplyMatrixVector(matrix_d, input_vectors[n]))
            if use_ending_state:
                state_vector = next_x
            else:
                state_vector = x
            trace.append(mp.mpStackVectors(input_vectors[n], mp.mpStackVectors(state_vector, \
                                                                               output_vector)))
            x = next_x
        return trace

    @staticmethod
    def extract_sequences(n_seq: Optional[Dict[str,mp.TTimeStampList]], u_seq: Optional[List[ \
        mp.TTimeStampList]], event_sequences: Dict[str,EventSequenceModel], \
            vector_sequences: Dict[str,VectorSequenceModel], input_labels: List[str]) \
                -> List[mp.TTimeStampList]:
        '''
        Extract the input sequence for the input labels from the combined information from named
        (nSeq) and unnamed (uSeq) sequences provided on the command line and the sequences defined
        in the model. Priority is given to sequences specified on the command line. Sequences that
        cannot be found by name are filled with the unnamed sequences.
        '''

        if u_seq is None:
            u_seq = []
        if n_seq is None:
            all_named = {}
        else:
            all_named = n_seq.copy()

        # add event sequences
        for e in event_sequences:
            all_named[e] = event_sequences[e].sequence()

        # add vector sequences.
        for v in vector_sequences:
            vsl = vector_sequences[v].extract_sequence_list()
            for k, vs in enumerate(vsl):
                all_named[v+str(k+1)] = vs

        res: List[mp.TTimeStampList] = []
        u_ind = 0
        for l in input_labels:
            if l in all_named:
                res.append(all_named[l])
            else:
                if u_ind >= len(u_seq):
                    raise MPMException("Insufficient input sequences specified")
                res.append(u_seq[u_ind])
                u_ind += 1
        return res

    @staticmethod
    def multiply_sequence(matrices: List[Union['MaxPlusMatrixModel',VectorSequenceModel]]) \
        ->Union['MaxPlusMatrixModel',VectorSequenceModel]:
        '''Multiply the sequence of matrices, possibly ending with a vector sequence.'''
        return reduce(lambda prod, mat: prod.multiply(mat), matrices)

    @staticmethod
    def from_dsl(dsl_string: str)-> Tuple[str,Dict[str,'MaxPlusMatrixModel'], \
                    Dict[str,VectorSequenceModel],Dict[str,EventSequenceModel]]:
        '''
        Parse dslString and extract model name, matrices, vector sequences and event sequences.
        Raise an exception if the parsing fails.
        '''

        factory = {}
        factory['Init'] = MaxPlusMatrixModel
        factory['AddRow'] = lambda mpm, r: mpm.add_row(r)
        factory['AddVector'] = lambda mpm, v: mpm.add_vector(v)
        factory['SetSequence'] = lambda mpm, s: mpm.set_sequence(s)
        factory['InitVectorSequence'] = VectorSequenceModel
        factory['InitEventSequence'] = EventSequenceModel
        factory['AddLabels'] = lambda m, labels: m.set_labels(labels)

        name, res_matrices, res_vector_sequences, res_event_sequences = \
            parseMPMDSL(dsl_string, factory)
        if name is None or res_matrices is None or res_vector_sequences is None or \
              res_event_sequences is None:
            raise MPMException("Failed to parse max-plus model")
        return name, res_matrices, res_vector_sequences, res_event_sequences


    def as_dsl(self, name: str, all_instances: Optional[Dict[str,Union['MaxPlusMatrixModel',\
                                VectorSequenceModel,EventSequenceModel]]] = None) -> str:

        '''
        Convert the receiver and all entities in optional allInstances to a string in the
        MPM Domain Specific Language using the provided name as the name for the model.
        '''

        # check if we want to output fractions or floats
        den = commonDenominatorMatrix(self._rows)
        pv = mpVectorToString if isComplex(den) else mpVectorToFractionString

        # create string writer for the output
        output = StringIO()
        if all_instances is None:
            output.write(f"max-plus model {name} : \nmatrices\nA = [\n")
            for r in self._rows:
                output.write(f"\t{pv(r)}\n")
            output.write("]\n")
        else:
            mats: List[str] = [i for i in all_instances if isinstance(all_instances[i], \
                                                                      MaxPlusMatrixModel)]
            v_sequences = [i for i in all_instances if isinstance(all_instances[i], \
                                                                  VectorSequenceModel)]
            e_sequences = [i for i in all_instances if isinstance(all_instances[i], \
                                                                  EventSequenceModel)]

            output.write(f"max-plus model {name} : \n")

            if len(mats) > 0:
                output.write("\nmatrices\n")
                for mat in mats:
                    mm: MaxPlusMatrixModel = all_instances[mat]  # type: ignore
                    if len(mm.labels()) > 0:
                        labels = f"({' '.join(mm.labels())})"
                    else:
                        labels = ''
                    output.write(f"{mat} {labels} = [\n")
                    for r in mm.rows():
                        output.write(f"\t{pv(r)}\n")
                    output.write("]\n")

            if len(v_sequences) > 0:
                output.write("\nvector sequences\n")
                for vs in v_sequences:
                    vsm: VectorSequenceModel = all_instances[vs]  # type: ignore
                    if len(vsm.labels()) > 0:
                        labels = f"({' '.join(vsm.labels())})"
                    else:
                        labels=''
                    output.write(f"{vs} {labels} = [\n")
                    for v in vsm.vectors():
                        output.write(f"\t{pv(v)}\n")
                    output.write("]\n")

            if len(e_sequences) > 0:
                output.write("\nevent sequences\n")
                for es in e_sequences:
                    esm: EventSequenceModel = all_instances[es]  # type: ignore
                    output.write(f"{es} = {pv(esm.sequence())}")

        result = output.getvalue()
        output.close()
        return result
