"""This module provides all functionality related to dataflow models."""

import copy
from functools import reduce
from io import StringIO
import sys
from typing import Tuple, Any, Set, Union, Dict, List, Optional
from fractions import Fraction
from dataflow.libsdfgrammar import parseSDFDSL
from dataflow.maxplus.starclosure import PositiveCycleException
from dataflow.maxplus.maxplus import TThroughputValue, mpThroughput, mpGeneralizedThroughput, \
      mpMatrixMinusScalar, mpStarClosure, mpMultiplyMatrices, mpMultiplyMatrixVector, \
      mpMinusInfVector, mpTransposeMatrix, mpZeroVector, mpMaxMatrices, mpMaxVectors, \
      mpScaleVector, mpSplitSequence, TTimeStamp, TTimeStampList, TMPVector, TMPMatrix
from dataflow.maxplus.algebra import MP_MINUSINFINITY
from dataflow.libmpm import MaxPlusMatrixModel

# constants
DEFAULT_ACTOR_EXECUTION_TIME = Fraction(1.0)
EXECUTION_TIME_SPEC_KEY = 'executionTime'
CONS_RATE_SPEC_KEY = 'consRate'
PROD_RATE_SPEC_KEY = 'prodRate'
INITIAL_TOKENS_SPEC_KEY = 'initialTokens'

TActorList = List[str]
TChannelList = List[str]
TActorSpecs = Dict[str,Any]
TChannelSpecs = Dict[str,Any]


def _split_matrix(matrix: TMPMatrix, n: int)->Tuple[TMPMatrix,TMPMatrix,TMPMatrix,TMPMatrix]:
    '''Split matrix into state-space A,B,C,D matrices assuming a state size n.'''
    matrix_a: TMPMatrix = []
    matrix_b: TMPMatrix = []
    for k in range(n):
        matrix_a.append(matrix[k][:n])
        matrix_b.append(matrix[k][n:])
    matrix_c: TMPMatrix = []
    matrix_d: TMPMatrix = []
    for k in range(n, len(matrix)):
        matrix_c.append(matrix[k][:n])
        matrix_d.append(matrix[k][n:])
    return (matrix_a, matrix_b, matrix_c, matrix_d)

class SDFException(Exception):
    """Exceptions related to this module"""

class SDFDeadlockException(SDFException):
    """Exception indicating a deadlock"""

class SDFInconsistentException(SDFException):
    """Exception indicating an inconsistent dataflow graph"""

class DataflowGraph:
    """Representation of dataflow graphs"""

    _repetition_vector: Optional[Dict[str,int]]
    _symbolic_vector: List[str]
    _inputs: List[str]
    _outputs: List[str]
    _actors_and_io: TActorList # note that the list _actorsAndIO includes the inputs and outputs
    _channels: TChannelList
    _actor_specs: Dict[str,TActorSpecs]
    _channel_specs: Dict[str,TChannelSpecs]
    _out_channels: Dict[str,Set[str]]
    _in_channels: Dict[str,Set[str]]
    _chan_producer: Dict[str,str]
    _chan_consumer: Dict[str,str]
    _input_signals: Dict[str,TTimeStampList]

    def __init__(self):
        # set of actors, including inputs and outputs!
        self._actors_and_io = []
        # set  of channels
        self._channels = []
        # dict actors -> (spec -> value)
        self._actor_specs = {}
        # dict actor -> set of channels
        self._out_channels = {}
        # dict actor -> set of channels
        self._in_channels = {}
        # dict chan -> producing actor
        self._chan_producer = {}
        # dict chan -> producing actor
        self._chan_consumer = {}
        # dict chan->(spec->value)
        self._channel_specs = {}
        # set of input 'actors'
        self._inputs = []
        # set of output 'actors'
        self._outputs = []

        # input signals
        self._input_signals = {}

        self._repetition_vector = None

        # initialized in _initialize_symbolic_time_stamps
        self._symbolic_time_stamp_size = -1


    def copy(self)->'DataflowGraph':
        ''' Return a new DataflowGraph as a copy of thw one the method of which
        is called. '''
        return copy.deepcopy(self)

    def actors(self)->TActorList:
        '''Return list of actors.'''
        return self._actors_and_io

    def channels(self)->TChannelList:
        '''Return list of channels.'''
        return self._channels

    def actors_without_inputs_outputs(self)->TActorList:
        '''Return only proper actors, not inputs or outputs.'''
        return [a for a in self._actors_and_io if not (a in self._inputs or a in self._outputs)]

    def inputs(self) -> List[str]:
        '''Return the inputs of the graph.'''
        return self._inputs

    def outputs(self)->List[str]:
        '''Return the outputs of the graph.'''
        return self._outputs

    def input_signals(self)->Dict[str,TTimeStampList]:
        '''Returns the inputs signals to the graph.'''
        return self._input_signals

    def consumption_rate(self, ch: str)->int:
        '''Get the consumption rate of the channel. Defaults to 1 if it is not specified.'''
        if ch in self._channel_specs:
            if CONS_RATE_SPEC_KEY in self._channel_specs[ch]:
                return self._channel_specs[ch][CONS_RATE_SPEC_KEY]
        return 1

    def production_rate(self, ch: str)->int:
        '''Get the production rate of the channel. Defaults to 1 if it is not specified.'''
        if ch in self._channel_specs:
            if PROD_RATE_SPEC_KEY in self._channel_specs[ch]:
                return self._channel_specs[ch][PROD_RATE_SPEC_KEY]
        return 1

    def repetitions(self, actor: str)->int:
        '''
        Determine repetition vector entry for actor. An
        SDFInconsistentException is raised if the graph is inconsistent.
        '''
        # Compute the repetition vector if needed
        if self._repetition_vector is None:
            rep_vec = self.repetition_vector()
            if isinstance(rep_vec, list):
                raise SDFInconsistentException("Dataflow graph is inconsistent")
            self._repetition_vector = rep_vec
        return self._repetition_vector[actor]

    def repetition_vector_sum(self)->int:
        '''Determine the sum of the repetitions of all actors, inputs and
        outputs. An SDFInconsistentException is raised if the graph is
        inconsistent.'''
        # Compute the repetition vector if needed
        if self._repetition_vector is None:
            rep_vec = self.repetition_vector()
            if isinstance(rep_vec, list):
                raise SDFInconsistentException("Dataflow graph is inconsistent")
            self._repetition_vector = rep_vec
        res: int = 0
        for a in self._actors_and_io:
            res = res + self._repetition_vector[a]
        return res

    def validate(self):
        '''
        Validate the dataflow model. Raises an exception on an invalid model. Returns nothing.
        '''
        unread_inputs = set(self._inputs).difference(set(self._actors_and_io))
        if len(unread_inputs) > 0:
            raise SDFException("Invalid model. The following inputs are not " \
                               f"read: {', '.join(unread_inputs)}.")
        unwritten_outputs = set(self._outputs).difference(set(self._actors_and_io))
        if len(unwritten_outputs) > 0:
            raise SDFException("Invalid model. The following outputs are not " \
                               f"written: {', '.join(unwritten_outputs)}.")

    def add_actor(self, a: str, specs: TActorSpecs):
        '''Add actor to the graph with specs.'''
        # invalidate the cached repetition vector
        self._repetition_vector = None
        # add actor if it doesn't exist
        if not a in self._actors_and_io:
            self._actors_and_io.append(a)
            self._actor_specs[a] = {}
        # add specs
        for s in specs:
            self._actor_specs[a][s] = specs[s]

    def add_channel(self, a: str, b: str, specs: TChannelSpecs):
        '''Add channel from actor or input a to actor or output b to the graph with specs.'''
        # invalidate the cached repetition vector
        self._repetition_vector = None

        ch_name: str
        if 'name' in specs:
            ch_name = specs['name']
        else:
            ch_name = self._new_channel_name()
        if not a in self._out_channels:
            self._out_channels[a] = set()
        self._out_channels[a].add(ch_name)
        if not b in self._in_channels:
            self._in_channels[b] = set()
        self._in_channels[b].add(ch_name)

        self._chan_producer[ch_name] = a
        self. _chan_consumer[ch_name] = b

        self._channels.append(ch_name)
        self._channel_specs[ch_name] = specs

    def add_input_port(self, i: str):
        '''Add input port i.'''
        self._repetition_vector = None
        # input ports should be in _actorsAndIO with execution time 0.0
        if not i in self._actors_and_io:
            self.add_actor(i, {EXECUTION_TIME_SPEC_KEY: Fraction(0.0)})
        else:
            self._actor_specs[i][EXECUTION_TIME_SPEC_KEY] = Fraction(0.0)
        self._inputs.append(i)

    def add_output_port(self, o: str):
        '''Add input port i.'''
        self._repetition_vector = None
        # output ports should be in actors
        if not o in self._actors_and_io:
            self.add_actor(o, {EXECUTION_TIME_SPEC_KEY: Fraction(0.0)})
        else:
            self._actor_specs[o][EXECUTION_TIME_SPEC_KEY] = Fraction(0.0)
        self._outputs.append(o)

    def add_input_signal(self, n: str, s: TTimeStampList):
        '''Add input signal with name n and sequences s.'''
        self._input_signals[n] = s

    def producer_of_channel(self, ch: str):
        '''Get the producer to channel ch.'''
        return self._chan_producer[ch]

    def consumer_of_channel(self, ch):
        '''Get the consumer from channel ch.'''
        return self._chan_consumer[ch]

    def _new_channel_name(self)->str:
        '''Generate a free channel name'''
        def fname(m):
            return 'ch'+str(m)
        k = 1
        while fname(k) in self._channel_specs:
            k += 1
        return fname(k)

    def execution_time_of_actor(self, a: str)->Fraction:
        '''Get the execution time of actor a'''
        if not EXECUTION_TIME_SPEC_KEY in self._actor_specs[a]:
            return DEFAULT_ACTOR_EXECUTION_TIME
        return self._actor_specs[a][EXECUTION_TIME_SPEC_KEY]

    def specs_of_actor(self, a: str)->Dict[str,Any]:
        '''Return the specs of actor a.'''
        return self._actor_specs[a]

    def number_of_initial_tokens_of_channel(self, ch: str)->int:
        '''Get the number of initial tokens on channel ch. Defaults to 0 if it is not specified.'''
        if not INITIAL_TOKENS_SPEC_KEY in self._channel_specs[ch]:
            return 0
        return self._channel_specs[ch][INITIAL_TOKENS_SPEC_KEY]

    def set_number_of_initial_tokens_of_channel(self, ch: str, it: int):
        '''Set the number of initial tokens on channel ch to it.'''
        self._channel_specs[ch][INITIAL_TOKENS_SPEC_KEY] = it

    def number_of_initial_tokens(self)->int:
        '''Get the total number of initial tokens in the graph.'''
        return reduce(lambda sum, ch: sum + self.number_of_initial_tokens_of_channel(ch), \
                      self._channels, 0)

    def number_of_inputs(self)->int:
        '''Get the number of inputs of the graph.'''
        return len(self._inputs)

    def number_of_outputs(self)->int:
        '''Get the number of outputs of the graph.'''
        return len(self._outputs)

    def number_of_inputs_in_iteration(self)->int:
        '''Get the total number of tokens consumed in one iteration.'''
        res: int = 0
        for i in self._inputs:
            res = res + self.repetitions(i)
        return res

    def channel_set(self)->Set[Tuple[str,str,int,int,int]]:
        '''Return a set with a tuple for all channels of the graph, containing the actor producing
           to the channel, the actor consuming from the channel, the number of initial tokens on
           the channel, the consumption rate of the channel and the production rate of the channel.
        '''
        result: Set[Tuple[str,str,int,int,int]] = set()
        for ch in self._channels:
            result.add(
                (
                    self._chan_producer[ch],
                    self._chan_consumer[ch],
                    self.number_of_initial_tokens_of_channel(ch),
                    self.consumption_rate(ch),
                    self.production_rate(ch)
                )
            )
        return result

    def in_channels(self, a: str)->Set[str]:
        '''Returns the set of channels that actor a consumes from.'''
        if a in self._in_channels:
            return self._in_channels[a]
        return set()

    def out_channels(self, a)->Set[str]:
        '''Returns the set of channels that actor a produces to.'''
        if a in self._out_channels:
            return self._out_channels[a]
        return set()

    def list_of_inputs_str(self)->str:
        '''Return a string representation of the list of inputs.'''
        return ', '.join(self._inputs)

    def list_of_outputs_str(self)->str:
        '''Return a string representation of the list of outputs.'''
        return ', '.join(self._outputs)

    def index_of_output(self, output: str)->int:
        '''Return the index of output in the list of outputs.'''
        return self._outputs.index(output)


    def state_element_labels(self)->List[str]:
        '''Return a list state element labels for the max-plus state-space representation
        of the graph.'''
        res: List[str] = []
        for ch in self._channels:
            for n in range(self.number_of_initial_tokens_of_channel(ch)):
                res.append(self._readable_initial_token_label(ch, n))
        return res

    def list_of_state_elements_str(self)->str:
        '''Return a string representation of the list of state element labels for the max-plus
           state-space representation of the graph.'''
        return ', '.join(self.state_element_labels())

    def _initial_token_label(self, ch: str, n: int)->str:
        '''Return a label for initial token number n on channel ch. Assumes n is between 1 and the
        number of initial tokens on the channel ch.'''
        if self.number_of_initial_tokens_of_channel(ch) == 1:
            return ch
        return ch+'_'+str(n+1)

    def _input_token_label(self, i: str, n: int)->str:
        '''Return a label for input token number n consumed from input i in one iteration.'''
        if self.repetitions(i) == 1:
            return i
        return i+'_'+str(n+1)

    def _output_token_label(self, o: str, n: int)->str:
        '''Return a label for output token number n produced to output o in one iteration.'''
        if self.repetitions(o) == 1:
            return o
        return o+'_'+str(n+1)

    def _actor_firing_label(self, a: str, n: int)->str:
        '''Return a label to represent firing number n of actor a in one iteration.'''
        if self.repetitions(a) == 1:
            return a
        return a+'_'+str(n+1)

    def _readable_initial_token_label(self, ch: str, n: int)->str:
        '''Return a label for initial token number n on channel ch using the actors connected to
        the channel. Assumes n is between 1 and the number of initial tokens on the channel ch.'''
        if self.number_of_initial_tokens_of_channel(ch) < 2:
            return f'{self._chan_producer[ch]}_{self._chan_consumer[ch]}'
        else:
            return f'{self._chan_producer[ch]}_{self. _chan_consumer[ch]}_{n+1}'

    def _initialize_symbolic_time_stamps(self):
        '''Determine the symbolic time stamp size and the symbolic vector labels.'''
        self._symbolic_time_stamp_size = self.number_of_initial_tokens() + \
            self.number_of_inputs_in_iteration()
        self._symbolic_vector = []
        for ch in self._channels:
            for n in range(self.number_of_initial_tokens_of_channel(ch)):
                self._symbolic_vector.append(self._initial_token_label(ch, n))
        for i in self._inputs:
            for n in range(self.repetitions(i)):
                self._symbolic_vector.append(self._input_token_label(i, n))

    def _symbolic_time_stamp_minus_infinity(self) -> TMPVector:
        '''Return a symbolic time stamp vector with all -inf elements.'''
        return [MP_MINUSINFINITY] * self._symbolic_time_stamp_size

    def symbolic_time_stamp(self, t: str)->TMPVector:
        '''Get the initial symbolic time stamp for the element labelled t, i.e., a corresponding
        unit vector.'''
        res = self._symbolic_time_stamp_minus_infinity()
        res[self._symbolic_vector.index(t)] = Fraction(0)
        return res

    def _symbolic_time_stamp_max(self, ts1: TMPVector, ts2: TMPVector)->TMPVector:
        '''Determine the maximum of two symbolic time stamp vectors.'''
        return mpMaxVectors(ts1, ts2)

    def _symbolic_time_stamp_scale(self, c: TTimeStamp, ts: TMPVector)->TMPVector:
        '''Scale the symbolic time stamp vector.'''
        return mpScaleVector(c, ts)

    def _symbolic_firing(self, a: str, n: int, timestamps:Dict[str,TMPVector])->bool:
        '''Attempt to fire actor a for the n'th time in the symbolic simulation of the graph.
        Return if the actor could be fired successfully or not.'''
        # FIXME: this method seems a bit inefficient.
        el: str = self._actor_firing_label(a, n)
        if el in timestamps:
            # This firing already has a determined time stamp
            return False
        # check if the dependencies are complete
        ts = self._symbolic_time_stamp_minus_infinity()
        for ch in self.in_channels(a):
            cr = self.consumption_rate(ch)
            cons = cr * (n+1)
            # determine how many initial tokens are consumed in the firing
            if self.number_of_initial_tokens_of_channel(ch) >= cons:
                it = cons
            else:
                it = self.number_of_initial_tokens_of_channel(ch)

            # determine the symbolic time stamps of the combined initial tokens consumed
            for k in range(it):
                ts = self._symbolic_time_stamp_max(ts, timestamps[self._initial_token_label(ch, k)])

            # determine how many tokens remain to be produced by the producing actor
            rem = cons - it
            b = self._chan_producer[ch]
            pr = self.production_rate(ch)
            for k in range(rem):
                # which firing produced token n ?
                m = k // pr
                elm = self._actor_firing_label(b, m)
                if not elm in timestamps:
                    # the production is not available yet
                    return False
                ts = self._symbolic_time_stamp_max(ts, self._symbolic_time_stamp_scale( \
                    self.execution_time_of_actor(b), timestamps[elm]))
        timestamps[el] = ts
        return True

    def _symbolic_completion_time(self, timestamps: Dict[str,TMPVector], a: str, n: int)->TMPVector:
        '''Determine the symbolic completion time of firing n of actor a.'''
        el = self._actor_firing_label(a, n)
        return self._symbolic_time_stamp_scale(self.execution_time_of_actor(a), timestamps[el])

    def state_space_matrices(self)->Tuple[TMPMatrix,Tuple[TMPMatrix,TMPMatrix,TMPMatrix,TMPMatrix]]:
        '''
        Compute the trace matrix and the state-space, A, B, C, and D, matrices.
        Returns a pair with
        - the trace matrix (H) with a row for every actor firing in an iteration
        - a four-tuple with the state-space matrices, A, B, C and D.
        A SDFDeadlockException is raised if the graph deadlocks.
        '''
        self._initialize_symbolic_time_stamps()
        timestamps: Dict[str,TMPVector] = {}
        # set the symbolic time stamps for all input tokens
        for i in self._inputs:
            for n in range(self.repetitions(i)):
                el = self._input_token_label(i, n)
                timestamps[el] = self.symbolic_time_stamp(el)
        # set the symbolic time stamps for all initial tokens on channels
        for ch in self._channels:
            for n in range(self.number_of_initial_tokens_of_channel(ch)):
                el = self._initial_token_label(ch, n)
                timestamps[el] = self.symbolic_time_stamp(el)
        # keep track of the number of firings of actors, inputs and outputs, initialize to 0
        actor_firings: Dict[str,int] = {}
        for a in self._actors_and_io:
            actor_firings[a] = 0

        old_len = len(timestamps)
        # while we need to compute more symbolic time stamps to complete an iteration...
        while len(timestamps)<self.repetition_vector_sum()+self.number_of_initial_tokens():
            # try to fire the actors, inputs and outputs one by one
            for a in self._actors_and_io:
                # only if it still needs firings to complete the iteration
                if actor_firings[a] < self.repetitions(a):
                    # determine the label
                    el = self._actor_firing_label(a, actor_firings[a])
                    # and if we don't yet have it
                    if not el in timestamps:
                        # try to execute the symbolic firing
                        if self._symbolic_firing(a, actor_firings[a], timestamps):
                            # it it succeeded, count the firing
                            actor_firings[a] += 1
            # check if we have made any progress
            if len(timestamps) == old_len:
                # if not, there is a deadlock
                raise SDFDeadlockException("The graph deadlocks.")
            old_len = len(timestamps)

        # determine the combined state-space matrix [A B; C D]
        matrix_a: TMPMatrix = []
        # first the rows corresponding to the new initial tokens
        for ch in self._channels:
            n_tokens = self.number_of_initial_tokens_of_channel(ch)
            cr = self.consumption_rate(ch)
            cons = cr*self.repetitions(self._chan_consumer[ch])
            for n in range(n_tokens):
                if n < n_tokens - cons:
                    # a shifting token
                    matrix_a.append(timestamps[self._initial_token_label(ch, n+cons)])
                else:
                    in_a = self._chan_producer[ch]
                    m = (n + cons - n_tokens ) // self.production_rate(ch)
                    matrix_a.append(self._symbolic_completion_time(timestamps, in_a, m))

        # then the rows corresponding to the outputs
        for o in self._outputs:
            for n in range(self.repetitions(o)):
                el = self._output_token_label(o, n)
                matrix_a.append(timestamps[el])

        # determine the trace matrix H
        matrix_h = []
        for a in self._actors_and_io:
            # if a is a proper actor (not an input or output)
            if not (a in self._inputs or a in self._outputs):
                for n in range(self.repetitions(a)):
                    el = self._actor_firing_label(a, n)
                    # add the symbolic starting time
                    matrix_h.append(timestamps[el])

        return matrix_h, _split_matrix(matrix_a, self.number_of_initial_tokens())

    def repetition_vector(self) -> Union[Dict[str,int],List[str]]:
        '''Determine the repetition vector of the graph. Returns None if the graph is
        inconsistent.'''
        def _find_integer_rates(comp: List[str], rates: Dict[str,Fraction]):
            '''Make the rates of all actors in set comp least integer values.'''
            # determine smallest scaling factor that makes fractional rates integer
            # in the set comp of actors
            factor = Fraction(1,1)
            for a in comp:
                # rate of a, scaled by current factor
                sc_rate = factor*rates[a]
                # if it is still fractional, increase factor by denominator
                if sc_rate.denominator > 1:
                    factor = factor * sc_rate.denominator
            # now scale all rates by factor to make them integer
            for a in comp:
                rates[a] = rates[a] * factor

        def _make_integer_rates(rates: Dict[str,Fraction])->Dict[str,int]:
            '''Convert integers represented as Fraction type to integer type.'''
            res = {}
            for a in rates:
                res[a] = rates[a].numerator
            return res

        def _get_ancestor_cycle(tree: Dict[str,str], node1: str, node2: str):

            def _find_common_ancestor(tree: Dict[str,str], node1: str, node2: str):

                def _is_ancestor(tree: Dict[str,str], node1: str, node2: str):
                    while node2 != node1 and node2 in tree:
                        node2 = tree[node2]
                    return node1 == node2

                while not _is_ancestor(tree, node1, node2):
                    node1 = tree[node1]

                return node1

            def _ancestor_path(tree: Dict[str,str], node: str, parent: str):
                res = []
                while node != parent:
                    res.append(node)
                    node = tree[node]
                res.append(parent)
                return res

            # start from node 1 upward and check if node is ancestor of node2, nc first one
            nc = _find_common_ancestor(tree, node1, node2)
            res = _ancestor_path(tree, node1, nc)
            res.reverse()
            res.extend(_ancestor_path(tree, node2, nc)[:-1])
            return res

        actors: List[str] = sorted(set(self._actors_and_io))

        # if there are no actors return trivial solution
        if len(actors)==0:
            return {}

        # keep computed fractional rates
        rates: Dict[str,Fraction] = {}

        # while there are more actors to explore
        # this loop is used once for every unconnected part of the graph
        while len(actors)>0:
            # next actor a
            a = next(iter(actors))

            # init to default rate
            rates[a] = Fraction(1,1)

            tree:Dict[str,str] = {}
            # actors to be processed, initialize with a
            proc: List[str] = list([a])
            # computed, initialize with a
            comp: List[str] = list([a])

            while len(proc)>0:
                # get next actor
                b = next(iter(proc))
                proc.remove(b)
                actors.remove(b)

                # does b have outgoing channels?
                if b in self._out_channels:
                    for c in self._out_channels[b]:
                        pr = self.production_rate(c)
                        co = self.consumption_rate(c)
                        ca = self._chan_consumer[c]
                        # determine the fractional rate of the connected actor ca
                        rate = rates[b] * pr / co
                        # if ca already has a rate
                        if ca in rates:
                            if not rate == rates[ca]:
                                # found inconsistent cycle
                                return _get_ancestor_cycle(tree, ca, b)
                        else:
                            # set b as parent of ca in the tree
                            tree[ca] = b
                            rates[ca] = rate
                            proc.append(ca)
                            comp.append(ca)
                # does b have incoming channels?
                if b in self._in_channels:
                    for c in self._in_channels[b]:
                        pr = self.production_rate(c)
                        co = self.consumption_rate(c)
                        ca = self._chan_producer[c]
                        rate = rates[b] * co / pr
                        if ca in rates:
                            if not rate == rates[ca]:
                                # found an inconsistent cycle
                                return _get_ancestor_cycle(tree, ca, b)
                        else:
                            tree[ca] = b
                            rates[ca] = rate
                            proc.append(ca)
                            comp.append(ca)
            _find_integer_rates(comp, rates)
        # convert fractional rates to integer rates
        return _make_integer_rates(rates)

    def throughput(self)->TThroughputValue:
        '''
        Compute throughput of the graph
        '''
        # compute state-space representation
        _, ssr = self.state_space_matrices()
        # compute throughput from the state matrix
        return mpThroughput(ssr[0])

    def throughput_output(self, output: str)->TThroughputValue:
        '''
        Compute throughput of the graph
        '''
        # compute state-space representation
        _, ssr = self.state_space_matrices()
        # compute throughput from the state matrix
        tp: List[TThroughputValue] = mpGeneralizedThroughput(ssr[0])
        i = self.index_of_output(output)
        # find the minimum in tp for all non-minus-infinity elements in row number i of the C matrix
        min_val: TThroughputValue = "infinite"
        c_r = ssr[2][i]
        for k, crv in enumerate(c_r):
            if not crv == MP_MINUSINFINITY:
                t: TThroughputValue = tp[k]
                if min_val=="infinite":
                    min_val = t
                elif not t=="infinite":
                    if t < min_val:
                        min_val = t
        return min_val


    def deadlock(self)->bool:
        '''
        Check if the dataflow graph deadlocks
        '''
        try:
            self.state_space_matrices()
        except SDFDeadlockException:
            return True
        return False

    def latency(self, x0: Optional[TMPVector], mu: Fraction)->TMPMatrix:
        '''Determine the mu-periodic latency of the dataflow graph. If x0 is provided, it is
        considered the initial state of the initial tokens. If it is not provided, a zero vector is
        assumed. The latency matrix is returned. I.e., the matrix: Lambda = (C ( A-mu )^{*} ( x0
        otimes [0 .inputs.. 0] oplus ( B - mu))  oplus D, where A, B, C, D are the sate-space
        matrices of the dataflow graph.
        '''

        _, matrix_m = self.state_space_matrices()
        (matrix_a, matrix_b, matrix_c, matrix_d) = \
            (matrix_m[0], matrix_m[1], matrix_m[2], matrix_m[3])

        if x0 is None:
            x0 = mpZeroVector(self.number_of_initial_tokens())

        # Compute the following latency matrix:
        # Lambda = (C ( A-mu )^{*} ( x0 \otimes [0 .inputs.. 0] oplus ( B - mu))  oplus D

        matrix_a_mu= mpMatrixMinusScalar(matrix_a, mu)
        try:
            sc_a_mu = mpStarClosure(matrix_a_mu)
        except PositiveCycleException:
            raise SDFException('The requested period mu is smaller than smallest period the' \
                               ' system can sustain. Therefore, it has no latency.') # pylint: disable=raise-missing-from
        c_sc_a_mu = mpMultiplyMatrices(matrix_c, sc_a_mu)
        x00 = mpMultiplyMatrices(mpTransposeMatrix([x0]), [mpZeroVector(len(self._inputs))])
        b_m_mu= mpMatrixMinusScalar(matrix_b, mu)
        x_00_b_m_mu = mpMaxMatrices(x00, b_m_mu)
        c_sc_a_mu_x_00_b_m_mu = mpMultiplyMatrices(c_sc_a_mu, x_00_b_m_mu)
        return mpMaxMatrices(c_sc_a_mu_x_00_b_m_mu, matrix_d)

    def generalized_latency(self, mu: Fraction):
        '''Determine the mu-periodic latency of the dataflow graph in the form of separate
        IO-Latency and initial state latency matrices. I.e., the matrix:
        Lambda_IO = (C ( A-mu )^{*} (B - mu)  oplus D,
        Lambda_x = (C ( A-mu )^{*}, where A, B, C, D are the sate-space matrices of the
        dataflow graph.
        '''

        _, matrix_m = self.state_space_matrices()
        (matrix_a, matrix_b, matrix_c, matrix_d) = \
            (matrix_m[0], matrix_m[1], matrix_m[2], matrix_m[3])

        # Lambda_IO =  = (C ( A-mu )^{*} (B - mu)  oplus D
        # Lambda_x =   (C ( A-mu )^{*}

        a_mu= mpMatrixMinusScalar(matrix_a, mu)
        try:
            sc_a_mu = mpStarClosure(a_mu)
        except PositiveCycleException:
            raise SDFException('The requested period mu is smaller than smallest period the '\
                               'system can sustain. Therefore, it has no latency.') # pylint: disable=raise-missing-from
        c_sc_a_mu = mpMultiplyMatrices(matrix_c, sc_a_mu)

        b_m_mu= mpMatrixMinusScalar(matrix_b, mu)
        c_sc_a_mu_b_m_mu = mpMultiplyMatrices(c_sc_a_mu, b_m_mu)
        return c_sc_a_mu, mpMaxMatrices(c_sc_a_mu_b_m_mu, matrix_d)

    def is_single_rate(self)->bool:
        '''Check if the graph is single-rate.'''
        for ch in self._channels:
            if ch in self._channel_specs:
                if PROD_RATE_SPEC_KEY in self._channel_specs[ch]:
                    if self._channel_specs[ch][PROD_RATE_SPEC_KEY] > 1:
                        return False
                if CONS_RATE_SPEC_KEY in self._channel_specs[ch]:
                    if self._channel_specs[ch][CONS_RATE_SPEC_KEY] > 1:
                        return False
        return True

    def convert_to_single_rate(self):
        '''Convert the graph to a single rate graph'''

        def _actor_name(a:str, n:int, rep_vec: Dict[str,int])->str:
            if rep_vec[a] == 1:
                return a
            return f'{a}{n+1}'

        def _add_channel(res: DataflowGraph, pa: str, ca: str, it: int):
            # add channel only if it does not yet exist
            for ch in res.channels():
                if res.producer_of_channel(ch) == pa and res.consumer_of_channel(ch) == ca and \
                    res.number_of_initial_tokens_of_channel(ch) == it:
                    return
            specs = {}
            if it > 0:
                specs[INITIAL_TOKENS_SPEC_KEY] = it
            res.add_channel(pa, ca, specs)

        # if it already is single rate, return a copy of the graph itself
        if self.is_single_rate():
            return self.copy()

        rep_vec = self.repetition_vector()
        if isinstance(rep_vec, list):
            raise SDFInconsistentException("Graph is inconsistent")

        res = DataflowGraph()

        for a in self.actors_without_inputs_outputs():
            if rep_vec[a] == 1:
                res.add_actor(a, self._actor_specs[a])
            else:
                for n in range(rep_vec[a]):
                    res.add_actor(_actor_name(a,n,rep_vec), self._actor_specs[a])

        for ch in self._channels:
            it = self.number_of_initial_tokens_of_channel(ch)
            pr = self._chan_producer[ch]
            co = self._chan_consumer[ch]
            p_rate = self.production_rate(ch)
            c_rate = self.consumption_rate(ch)
            for n in range(rep_vec[pr] * p_rate):
                # token n is produced by actor firing n // pRate
                pa = _actor_name(pr, n//p_rate, rep_vec)
                # token is consumed by actor firing (n+it) // cRate
                ca = _actor_name(co, ((n+it) // c_rate) % rep_vec[co], rep_vec)
                # number of it ((n+it) // cRate) // repVec[co]
                nit = ((n+it) // c_rate) // rep_vec[co]
                _add_channel(res, pa, ca, nit)

        for i in self._inputs:
            for n in range(rep_vec[i]):
                res.add_input_port(_actor_name(i, n, rep_vec))

        for o in self._outputs:
            for n in range(rep_vec[o]):
                res.add_output_port(_actor_name(o, n, rep_vec))

        for i, s_i in self._input_signals.items():
            if not i in self._inputs:
                res.add_input_signal(i, s_i)
            else:
                seqs = mpSplitSequence(s_i, rep_vec[i])
                for n in range(rep_vec[i]):
                    res.add_input_signal(_actor_name(i, n, rep_vec), seqs[n])

        return res

    def determine_trace(self, ni: int, x0: Optional[TMPVector]=None, \
                        input_override: Optional[Dict[str,Union[TTimeStampList,str]]]=None) -> \
                            Tuple[List[TTimeStampList],List[TTimeStampList],List[TTimeStampList], \
                            List[Fraction]]:
        '''Determine execution trace for the dataflow graph.
        The trace is ni iterations long.
        x0 is an optional initial state for the execution. If it is not provided, initial tokens
        are assumed to be available at time 0.
        inputOverride, is optionally used to provide input sequences to replace the ones in the
        model.
        Inputs that are neither specified in the model, nor in the override, are assumed to provide
        all input tokens with times tamps -inf.
        Returns a tuple with the following elements
        input traces, the output traces, all firing start times, all firing durations.
        '''

        # determine the state-space model an the trace matrix
        matrix_h, state_space_matrices = self.state_space_matrices()

        # compute vector trace from state-space matrices
        matrices = {'A': MaxPlusMatrixModel.from_matrix(state_space_matrices[0]), \
                    'B': MaxPlusMatrixModel.from_matrix(state_space_matrices[1]), \
                    'C': MaxPlusMatrixModel.from_matrix(state_space_matrices[2]), \
                    'D': MaxPlusMatrixModel.from_matrix(state_space_matrices[3]) }

        state_vector_size = self.number_of_initial_tokens()
        repetition_vector: Dict[str,int] = self.repetition_vector() # type: ignore
        input_vector_size = reduce(lambda sum, i: sum+repetition_vector[i], self.inputs(), 0)

        if x0 is None:
            x0 = mpZeroVector(matrices['A'].number_of_columns())

        inp_sig = self.input_signals()
        inputs: List[TTimeStampList] = []
        for s in self.inputs():
            if input_override and s in input_override:
                s: str
                if isinstance(input_override[s], list):
                    ios_l: TTimeStampList = input_override[s]  # type: ignore
                    # the input is given as an list of time stamps.
                    # split it according to the inputs within one graph iteration
                    inputs.extend(mpSplitSequence(ios_l, self.repetitions(s)))
                else:
                    # the input is given as a name referring to an input sequence specified
                    #  in the model
                    ios_s: str = input_override[s]  # type: ignore
                    if input_override[s] not in inp_sig:
                        raise SDFException(f"Unknown event sequence: {input_override[s]}.")
                    inputs.extend(mpSplitSequence(inp_sig[ios_s], self.repetitions(s)))
            else:
                # the input is not specified in override
                if s in inp_sig:
                    # it is defined in the model, use it
                    inputs.extend(mpSplitSequence(inp_sig[s], self.repetitions(s)))
                else:
                    # it is not specified at all, use an event sequence with minus infinity
                    inputs.extend([mpMinusInfVector(ni)] * self.repetitions(s))

        # Compute the vector trace
        vt = MaxPlusMatrixModel.vector_trace(matrices, x0, ni, inputs)

        input_traces = [v[0:input_vector_size] for v in vt]
        output_traces = [v[input_vector_size+state_vector_size:] for v in vt]

        # reorder the vectors so that the state elements come first, followed by inputs
        ssvt = [v[input_vector_size:state_vector_size+input_vector_size]+ \
                v[0:input_vector_size] for v in vt]
        # compute the firing starting times using the trace matrix H
        firing_starts = [mpMultiplyMatrixVector(matrix_h, s)  for s in ssvt]
        # collect the firing durations
        firing_durations= [self.execution_time_of_actor(a) for a in \
                           self.actors_without_inputs_outputs()]

        return input_traces, output_traces, firing_starts, firing_durations

    def determine_trace_zero_based(self, ni:int, x0: Optional[TMPVector]=None) -> \
        Tuple[List[str],List[str],List[TTimeStampList],List[str],List[TTimeStampList],\
              List[TTimeStampList],List[Fraction]]:
        '''Determine a trace with ni iterations, assuming that actors cannot fire before time 0.
        Optional x0 can be used to specify an initial state for the graph.
        Returns a tuple with the following elements
        actors, inputs, input traces, outputs, the output traces, all firing start times, all
        firing durations.
        '''

        # determine trace assuming actors do not start before time 0
        # clone the graph to modify it.
        matrix_g = self.copy()

        # create artificial inputs to actors to constraint their firings.
        for a in matrix_g.actors_without_inputs_outputs():
            inp_name = f'_zb_{a}'
            matrix_g.add_input_port(inp_name)
            matrix_g.add_channel(inp_name, a, {})
            # set the input sequence to the new channel with tokens with time stamp 0 to
            # prevent it from firing earlier.
            matrix_g.add_input_signal(inp_name, [Fraction(0)] * ni)

        input_traces, output_traces, firing_starts, firing_durations = \
            matrix_g.determine_trace(ni, x0)

        # suppress the artificial inputs
        num = len(matrix_g.actors_without_inputs_outputs())
        def reduce_real_inputs(l):
            return l[:-num]
        real_input_traces = list(map(reduce_real_inputs, input_traces))

        return matrix_g.actors_without_inputs_outputs(), (matrix_g.inputs())[:-num],\
             real_input_traces, matrix_g.outputs(), output_traces, firing_starts, firing_durations

    def as_dsl(self, name: str)->str:
        '''Convert the model to a string representation in the domain specific language using
        the provided name.'''

        def _actor_specs(a: str, actors_with_spec: Set[str])->str:
            # if the specs of actor a have already been added to some instance of the actor,
            # return an empty string
            if a in actors_with_spec:
                return ''
            # mark that the specs have been written
            actors_with_spec.add(a)
            # if a has no specs
            if not a in self._actor_specs:
                return ''
            # if a has specs, but no execution time spec
            if not EXECUTION_TIME_SPEC_KEY in self._actor_specs[a]:
                return ''
            # otherwise return the execution time spec for the DSL
            return f'[{self._actor_specs[a][EXECUTION_TIME_SPEC_KEY]}]'

        def _channel_specs(ch: str)->str:
            '''Generate the channel spec for the channel.'''
            specs = []
            if ch in self._channel_specs:
                if CONS_RATE_SPEC_KEY in self._channel_specs[ch]:
                    specs.append(' consumption rate: ' \
                                 f'{self._channel_specs[ch][CONS_RATE_SPEC_KEY]} ')
                if PROD_RATE_SPEC_KEY in self._channel_specs[ch]:
                    specs.append(' production rate: ' \
                                 f'{self._channel_specs[ch][PROD_RATE_SPEC_KEY]} ')
                if INITIAL_TOKENS_SPEC_KEY in self._channel_specs[ch]:
                    specs.append(' initial tokens: ' \
                                 f'{self._channel_specs[ch][INITIAL_TOKENS_SPEC_KEY]} ')
            return ';'.join(specs)

        # create string writer for the output
        output = StringIO()
        output.write(f"dataflow graph {name} {{\n")

        if len(self._inputs)>0:
            output.write('\tinputs ')
            output.write(', '.join(self._inputs))
            output.write('\n')

        if len(self._outputs)>0:
            output.write('\toutputs ')
            output.write(', '.join(self._outputs))
            output.write('\n')

        actors_with_spec = set()
        for ch in self._channels:
            pr = self._chan_producer[ch]
            co = self._chan_consumer[ch]
            output.write(f'\t{pr}{_actor_specs(pr, actors_with_spec)} ')
            output.write(f' ----{_channel_specs(ch)}----> ')
            output.write(f'{co}{_actor_specs(co, actors_with_spec)}\n')
        output.write("}\n")

        if len(self._input_signals) > 0:
            output.write('\ninput signals\n\n')
            for inp, inp_sig in self._input_signals.items():
                input_signal_ratio_list = "["+", ".join([f"{i}" for i in inp_sig])+"]"
                output.write(f'{inp} = {input_signal_ratio_list}\n')

        result = output.getvalue()
        output.close()
        return result

    @staticmethod
    def from_dsl(dsl_string)->Tuple[str,'DataflowGraph']:
        '''
        Parse the model provides as a string.
        Returns if successful a pair with the name of the model and the constructed instance
        of `DataflowGraph`
        '''

        factory = {}
        factory['Init'] = DataflowGraph
        factory['AddActor'] = lambda sdf, a, specs: sdf.add_actor(a, specs)
        factory['AddChannel'] = lambda sdf, a1, a2, specs: sdf.add_channel(a1, a2, specs)
        factory['AddInputPort'] = lambda sdf, i: sdf.add_input_port(i)
        factory['AddOutputPort'] = lambda sdf, i: sdf.add_output_port(i)
        factory['AddInputSignal'] = lambda sdf, n, s: sdf.add_input_signal(n, s)
        result = parseSDFDSL(dsl_string, factory)
        if result[0] is None:
            sys.exit(1)
        return result  # type: ignore


    def __str__(self):
        return f"({self._actors_and_io}, {self._channels}, {self._inputs}, {self._outputs})"
