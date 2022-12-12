import copy
from typing import Literal, Tuple,Any,Set,Union
from functools import reduce
from io import StringIO
from typing import Dict, List, Optional
from dataflow.libsdfgrammar import parseSDFDSL
from dataflow.maxplus.starclosure import PositiveCycleException
from dataflow.maxplus.maxplus import mpThroughput, mpMatrixMinusScalar, mpStarClosure, mpMultiplyMatrices, mpMultiplyMatrixVector, mpMinusInfVector, mpTransposeMatrix, mpZeroVector, mpMaxMatrices, mpMaxVectors, mpScaleVector, mpSplitSequence, TTimeStamp, TTimeStampList, TMPVector, TMPMatrix
from dataflow.maxplus.algebra import MP_MINUSINFINITY
from dataflow.libmpm import MaxPlusMatrixModel
from fractions import Fraction



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

def _splitMatrix(M: TMPMatrix, n: int)->Tuple[TMPMatrix,TMPMatrix,TMPMatrix,TMPMatrix]:
    '''Split matrix into state-space A,B,C,D matrices assuming a state size n.'''
    A: TMPMatrix = []
    B: TMPMatrix = []
    for k in range(n):
        A.append(M[k][:n])
        B.append(M[k][n:])
    C: TMPMatrix = []
    D: TMPMatrix = []
    for k in range(n, len(M)):
        C.append(M[k][:n])
        D.append(M[k][n:])
    return (A, B, C, D)

class SDFException(Exception):
    pass

class SDFDeadlockException(SDFException):
    pass

class SDFInconsistentException(SDFException):
    pass

class DataflowGraph(object):

    _repetitionVector: Optional[Dict[str,int]]
    _symbolicVector: List[str]
    _inputs: List[str]
    _outputs: List[str]
    _actorsAndIO: TActorList # note that the list _actorsAndIO includes the inputs and outputs
    _channels: TChannelList
    _actorSpecs: Dict[str,TActorSpecs]
    _channelSpecs: Dict[str,TChannelSpecs]
    _outChannels: Dict[str,Set[str]]
    _inChannels: Dict[str,Set[str]]
    _chanProducer: Dict[str,str]
    _chanConsumer: Dict[str,str]
    _inputSignals: Dict[str,TTimeStampList]

    def __init__(self):
        # set of actors, including inputs and outputs!
        self._actorsAndIO = list()
        # set  of channels
        self._channels = list()
        # dict actors -> (spec -> value)
        self._actorSpecs = dict()
        # dict actor -> set of channels
        self._outChannels = dict()
        # dict actor -> set of channels
        self._inChannels = dict()
        # dict chan -> producing actor
        self._chanProducer = dict()
        # dict chan -> producing actor
        self._chanConsumer = dict()
        # dict chan->(spec->value)
        self._channelSpecs = dict()
        # set of input 'actors'
        self._inputs = list()
        # set of output 'actors'
        self._outputs = list()

        # input signals
        self._inputSignals = dict()

        self._repetitionVector = None

    def copy(self)->'DataflowGraph':
        ''' Return a new DataflowGraph as a copy of thw one the method of which
        is called. '''
        return copy.deepcopy(self)
        
    def actors(self)->TActorList:
        '''Return list of actors.'''
        return self._actorsAndIO

    def channels(self)->TChannelList:
        '''Return list of channels.'''
        return self._channels

    def actorsWithoutInputsOutputs(self)->TActorList:
        return [a for a in self._actorsAndIO if not (a in self._inputs or a in self._outputs)]

    def inputs(self) -> List[str]:
        '''Return the inputs of the graph.'''
        return self._inputs

    def outputs(self)->List[str]:
        '''Return the outputs of the graph.'''
        return self._outputs

    def inputSignals(self)->Dict[str,TTimeStampList]:
        '''Returns the inputs signals to the graph.'''
        return self._inputSignals

    def consumptionRate(self, ch: str)->int:
        '''Get the consumption rate of the channel. Defaults to 1 if it is not specified.'''
        if ch in self._channelSpecs:
            if CONS_RATE_SPEC_KEY in self._channelSpecs[ch]:
                return self._channelSpecs[ch][CONS_RATE_SPEC_KEY]
        return 1

    def productionRate(self, ch: str)->int:
        '''Get the production rate of the channel. Defaults to 1 if it is not specified.'''
        if ch in self._channelSpecs:
            if PROD_RATE_SPEC_KEY in self._channelSpecs[ch]:
                return self._channelSpecs[ch][PROD_RATE_SPEC_KEY]
        return 1

    def repetitions(self, actor: str)->int:
        '''
        Determine repetition vector entry for actor. An SDFInconsistentException is raised if the graph is inconsistent.
        '''
        # Compute the repetition vector if needed
        if self._repetitionVector is None:
            repVec = self.repetitionVector()
            if isinstance(repVec, list):
                raise SDFInconsistentException("Dataflow graph is inconsistent")
            self._repetitionVector = repVec 
        return self._repetitionVector[actor]

    def repetitionVectorSum(self)->int:
        '''Determine the sum of the repetitions of all actors, inputs and outputs. An SDFInconsistentException is raised if the graph is inconsistent.'''
        # Compute the repetition vector if needed
        if self._repetitionVector is None:
            repVec = self.repetitionVector()
            if isinstance(repVec, list):
                raise SDFInconsistentException("Dataflow graph is inconsistent")
            self._repetitionVector = repVec
        res: int = 0
        for a in self._actorsAndIO:
            res = res + self._repetitionVector[a]
        return res

    def validate(self):
        '''
        Validate the dataflow model. Raises an exception on an invalid model. Returns nothing.
        '''
        unreadInputs = set(self._inputs).difference(set(self._actorsAndIO))
        if len(unreadInputs) > 0:
            raise SDFException('Invalid model. The following inputs are not read: {}.'.format('. '.join(unreadInputs)))
        unwrittenOutputs = set(self._outputs).difference(set(self._actorsAndIO))
        if len(unwrittenOutputs) > 0:
            raise SDFException('Invalid model. The following outputs are not written: {}.'.format('. '.join(unwrittenOutputs)))

    def addActor(self, a: str, specs: TActorSpecs):
        '''Add actor to the graph with specs.'''
        # invalidate the cached repetition vector
        self._repetitionVector = None
        # add actor if it doesn't exist
        if not a in self._actorsAndIO:
            self._actorsAndIO.append(a)
            self._actorSpecs[a] = dict()
        # add specs
        for s in specs:
            self._actorSpecs[a][s] = specs[s]

    def addChannel(self, a: str, b: str, specs: TChannelSpecs):
        '''Add channel from actor or input a to actor or output b to the graph with specs.'''
        # invalidate the cached repetition vector
        self._repetitionVector = None
        
        chName: str
        if 'name' in specs:
            chName = specs['name']
        else:
            chName = self._newChannelName()
        if not a in self._outChannels:
            self._outChannels[a] = set()
        self._outChannels[a].add(chName)
        if not b in self._inChannels:
            self._inChannels[b] = set()
        self._inChannels[b].add(chName)

        self._chanProducer[chName] = a
        self. _chanConsumer[chName] = b

        self._channels.append(chName)
        self._channelSpecs[chName] = specs

    def addInputPort(self, i: str):
        '''Add input port i.'''
        self._repetitionVector = None
        # input ports should be in _actorsAndIO with execution time 0.0
        if not i in self._actorsAndIO:
            self.addActor(i, {EXECUTION_TIME_SPEC_KEY: Fraction(0.0)})
        else:
            self._actorSpecs[i][EXECUTION_TIME_SPEC_KEY] = Fraction(0.0)
        self._inputs.append(i)

    def addOutputPort(self, o: str):
        '''Add input port i.'''
        self._repetitionVector = None
        # output ports should be in actors
        if not o in self._actorsAndIO:
            self.addActor(o, {EXECUTION_TIME_SPEC_KEY: Fraction(0.0)})
        else:
            self._actorSpecs[o][EXECUTION_TIME_SPEC_KEY] = Fraction(0.0)
        self._outputs.append(o)

    def addInputSignal(self, n: str, s: TTimeStampList):
        '''Add input signal with name n and sequences s.'''
        self._inputSignals[n] = s

    def producerOfChannel(self, ch: str):
        '''Get the producer to channel ch.'''
        return self._chanProducer[ch]

    def consumerOfChannel(self, ch):
        '''Get the consumer from channel ch.'''
        return self._chanConsumer[ch]

    def _newChannelName(self)->str:
        '''Generate a free channel name'''
        fname = lambda m: 'ch'+str(m)
        k = 1
        while fname(k) in self._channelSpecs:
            k += 1
        return fname(k)

    def executionTimeOfActor(self, a: str)->Fraction:
        '''Get the execution time of actor a'''
        if not EXECUTION_TIME_SPEC_KEY in self._actorSpecs[a]:
            return DEFAULT_ACTOR_EXECUTION_TIME
        return self._actorSpecs[a][EXECUTION_TIME_SPEC_KEY]

    def specsOfActor(self, a: str)->Dict[str,Any]:
        '''Return the specs of actor a.'''
        return self._actorSpecs[a]

    def numberOfInitialTokensOfChannel(self, ch: str)->int:
        '''Get the number of initial tokens on channel ch. Defaults to 0 if it is not specified.'''
        if not INITIAL_TOKENS_SPEC_KEY in self._channelSpecs[ch]:
            return 0
        return self._channelSpecs[ch][INITIAL_TOKENS_SPEC_KEY]

    def setNumberOfInitialTokensOfChannel(self, ch: str, it: int):
        '''Set the number of initial tokens on channel ch to it.'''
        self._channelSpecs[ch][INITIAL_TOKENS_SPEC_KEY] = it

    def numberOfInitialTokens(self)->int:
        '''Get the total number of initial tokens in the graph.'''
        return reduce(lambda sum, ch: sum + self.numberOfInitialTokensOfChannel(ch), self._channels, 0)

    def numberOfInputs(self)->int:
        '''Get the number of inputs of the graph.'''
        return len(self._inputs)

    def numberOfOutputs(self)->int:
        '''Get the number of outputs of the graph.'''
        return len(self._outputs)

    def numberOfInputsInIteration(self)->int:
        '''Get the total number of tokens consumed in one iteration.'''
        res: int = 0
        for i in self._inputs:
            res = res + self.repetitions(i)
        return res
    
    def channelSet(self)->Set[Tuple[str,str,int,int,int]]:
        '''Return a set with a tuple for all channels of the graph, containing the actor producing to the channel, the actor consuming from the channel, the number of initial tokens on the channel, the consumption rate of the channel and the production rate of the channel.'''
        result: Set[Tuple[str,str,int,int,int]] = set()
        for ch in self._channels:
            result.add(
                (
                    self._chanProducer[ch], 
                    self._chanConsumer[ch], 
                    self.numberOfInitialTokensOfChannel(ch),
                    self.consumptionRate(ch),
                    self.productionRate(ch)
                )
            )
        return result

    def inChannels(self, a: str)->Set[str]:
        '''Returns the set of channels that actor a consumes from.'''
        if a in self._inChannels:
            return self._inChannels[a]
        return set()

    def outChannels(self, a)->Set[str]:
        '''Returns the set of channels that actor a produces to.'''
        if a in self._outChannels:
            return self._outChannels[a]
        return set()

    def listOfInputsStr(self)->str:
        '''Return a string representation of the list of inputs.'''
        return ', '.join(self._inputs)

    def listOfOutputsStr(self)->str:
        '''Return a string representation of the list of outputs.'''
        return ', '.join(self._outputs)

    def stateElementLabels(self)->List[str]:
        '''Return a list state element labels for the max-plus state-space representation of the graph.'''
        res: List[str] = []
        for ch in self._channels:
            for n in range(self.numberOfInitialTokensOfChannel(ch)):
                res.append(self._readableInitialTokenLabel(ch, n))
        return res

    def listOfStateElementsStr(self)->str:
        '''Return a string representation of the list of state element labels for the max-plus state-space representation of the graph.'''
        return ', '.join(self.stateElementLabels())

    def _initialTokenLabel(self, ch: str, n: int)->str:
        '''Return a label for initial token number n on channel ch. Assumes n is between 1 and the number of initial tokens on the channel ch.'''
        if self.numberOfInitialTokensOfChannel(ch) == 1:
            return ch
        return ch+'_'+str(n+1)

    def _inputTokenLabel(self, i: str, n: int)->str:
        '''Return a label for input token number n consumed from input i in one iteration.'''
        if self.repetitions(i) == 1:
            return i
        return i+'_'+str(n+1)

    def _outputTokenLabel(self, o: str, n: int)->str:
        '''Return a label for output token number n produced to output o in one iteration.'''
        if self.repetitions(o) == 1:
            return o
        return o+'_'+str(n+1)

    def _actorFiringLabel(self, a: str, n: int)->str:
        '''Return a label to represent firing number n of actor a in one iteration.'''
        if self.repetitions(a) == 1:
            return a
        return a+'_'+str(n+1)

    def _readableInitialTokenLabel(self, ch: str, n: int)->str:
        '''Return a label for initial token number n on channel ch using the actors connected to the channel. Assumes n is between 1 and the number of initial tokens on the channel ch.'''
        if self.numberOfInitialTokensOfChannel(ch) < 2:
            return '{}_{}'.format(self._chanProducer[ch], self. _chanConsumer[ch])
        else:
            return '{}_{}_{}'.format(self._chanProducer[ch], self. _chanConsumer[ch], n+1)

    def _initializeSymbolicTimeStamps(self):
        '''Determine the symbolic time stamp size and the symbolic vector labels.'''
        self._symbolicTimeStampSize = self.numberOfInitialTokens() + self.numberOfInputsInIteration()
        self._symbolicVector = []
        for ch in self._channels:
            for n in range(self.numberOfInitialTokensOfChannel(ch)):
                self._symbolicVector.append(self._initialTokenLabel(ch, n))
        for i in self._inputs:
            for n in range(self.repetitions(i)):
                self._symbolicVector.append(self._inputTokenLabel(i, n))

    def _symbolicTimeStampMinusInfinity(self) -> TMPVector:
        '''Return a symbolic time stamp vector with all -inf elements.'''
        return [MP_MINUSINFINITY] * self._symbolicTimeStampSize

    def symbolicTimeStamp(self, t: str)->TMPVector:
        '''Get the initial symbolic time stamp for the element labelled t, i.e., a corresponding unit vector.'''
        res = self._symbolicTimeStampMinusInfinity()
        res[self._symbolicVector.index(t)] = Fraction(0)
        return res

    def _symbolicTimeStampMax(self, ts1: TMPVector, ts2: TMPVector)->TMPVector:
        '''Determine the maximum of two symbolic time stamp vectors.'''
        return mpMaxVectors(ts1, ts2)

    def _symbolicTimeStampScale(self, c: TTimeStamp, ts: TMPVector)->TMPVector:
        '''Scale the symbolic time stamp vector.'''
        return mpScaleVector(c, ts)

    def _symbolicFiring(self, a: str, n: int, timestamps:Dict[str,TMPVector])->bool:
        '''Attempt to fire actor a for the n'th time in the symbolic simulation of the graph. Return if the actor could be fired successfully or not.'''
        # TODO: this method seems a bit inefficient.
        el: str = self._actorFiringLabel(a, n)
        if el in timestamps:
            # This firing already has a determined time stamp
            return False
        # check if the dependencies are complete
        ts = self._symbolicTimeStampMinusInfinity()
        for ch in self.inChannels(a):
            cr = self.consumptionRate(ch)
            cons = cr * (n+1)
            # determine how many initial tokens are consumed in the firing
            if self.numberOfInitialTokensOfChannel(ch) >= cons:
                it = cons
            else:
                it = self.numberOfInitialTokensOfChannel(ch)
            
            # determine the symbolic time stamps of the combined initial tokens consumed
            for k in range(it):
                ts = self._symbolicTimeStampMax(ts, timestamps[self._initialTokenLabel(ch, k)])

            # determine how many tokens remain to be produced by the producing actor
            rem = cons - it
            b = self._chanProducer[ch]
            pr = self.productionRate(ch)
            for k in range(rem):
                # which firing produced token n ?
                m = k // pr
                elm = self._actorFiringLabel(b, m)
                if not elm in timestamps:
                    # the production is not available yet
                    return False
                ts = self._symbolicTimeStampMax(ts, self._symbolicTimeStampScale(self.executionTimeOfActor(b), timestamps[elm]))
        timestamps[el] = ts
        return True

    def _symbolicCompletionTime(self, timestamps: Dict[str,TMPVector], a: str, n: int)->TMPVector:
        '''Determine the symbolic completion time of firing n of actor a.'''
        el = self._actorFiringLabel(a, n)
        return self._symbolicTimeStampScale(self.executionTimeOfActor(a), timestamps[el])

    def stateSpaceMatrices(self)->Tuple[TMPMatrix,Tuple[TMPMatrix,TMPMatrix,TMPMatrix,TMPMatrix]]:
        '''
        Compute the trace matrix and the state-space, A, B, C, and D, matrices.
        Returns a pair with 
        - the trace matrix (H) with a row for every actor firing in an iteration
        - a four-tuple with the state-space matrices, A, B, C and D.
        A SDFDeadlockException is raised if the graph deadlocks.
        '''
        self._initializeSymbolicTimeStamps() 
        timestamps: Dict[str,TMPVector] = dict()
        # set the symbolic time stamps for all input tokens
        for i in self._inputs:
            for n in range(self.repetitions(i)):
                el = self._inputTokenLabel(i, n)
                timestamps[el] = self.symbolicTimeStamp(el)
        # set the symbolic time stamps for all initial tokens on channels
        for ch in self._channels:
            for n in range(self.numberOfInitialTokensOfChannel(ch)):
                el = self._initialTokenLabel(ch, n)
                timestamps[el] = self.symbolicTimeStamp(el)
        # keep track of the number of firings of actors, inputs and outputs, initialize to 0
        actorFirings: Dict[str,int] = dict()
        for a in self._actorsAndIO:
            actorFirings[a] = 0
        
        oldLen = len(timestamps)
        # while we need to compute more symbolic time stamps to complete an iteration...
        while len(timestamps)<self.repetitionVectorSum()+self.numberOfInitialTokens():
            # try to fire the actors, inputs and outputs one by one
            for a in self._actorsAndIO:
                # only if it still needs firings to complete the iteration
                if actorFirings[a] < self.repetitions(a):
                    # determine the label
                    el = self._actorFiringLabel(a, actorFirings[a])
                    # and if we don't yet have it
                    if not el in timestamps:
                        # try to execute the symbolic firing
                        if self._symbolicFiring(a, actorFirings[a], timestamps):
                            # it it succeeded, count the firing
                            actorFirings[a] += 1
            # check if we have made any progress
            if len(timestamps) == oldLen:
                # if not, there is a deadlock
                raise SDFDeadlockException("The graph deadlocks.")
            oldLen = len(timestamps)
        
        # determine the combined state-space matrix [A B; C D]
        A: TMPMatrix = []
        # first the rows corresponding to the new initial tokens
        for ch in self._channels:
            nTokens = self.numberOfInitialTokensOfChannel(ch)
            cr = self.consumptionRate(ch)
            cons = cr*self.repetitions(self._chanConsumer[ch])
            for n in range(nTokens):
                if n < nTokens - cons:
                    # a shifting token
                    A.append(timestamps[self._initialTokenLabel(ch, n+cons)])
                else:
                    inA = self._chanProducer[ch]
                    m = (n + cons - nTokens ) // self.productionRate(ch)
                    A.append(self._symbolicCompletionTime(timestamps, inA, m))

        # then the rows corresponding to the outputs
        for o in self._outputs:
            for n in range(self.repetitions(o)):
                el = self._outputTokenLabel(o, n)
                A.append(timestamps[el])

        # determine the trace matrix H
        H = []
        for a in self._actorsAndIO:
            # if a is a proper actor (not an input or output)
            if not (a in self._inputs or a in self._outputs):
                for n in range(self.repetitions(a)):
                    el = self._actorFiringLabel(a, n)
                    # add the symbolic starting time
                    H.append(timestamps[el])

        return H, _splitMatrix(A, self.numberOfInitialTokens())

    def repetitionVector(self) -> Union[Dict[str,int],List[str]]:
        '''Determine the repetition vector of the graph. Returns None if the graph is inconsistent.'''
        def _findIntegerRates(comp: List[str], rates: Dict[str,Fraction]):
            '''Make the rates of all actors in set comp least integer values.'''
            # determine smallest scaling factor that makes fractional rates integer in the set comp of actors
            factor = Fraction(1,1)
            for a in comp:
                # rate of a, scaled by current factor
                scRate = factor*rates[a]
                # if it is still fractional, increase factor by denominator
                if scRate.denominator > 1:
                    factor = factor * scRate.denominator
            # now scale all rates by factor to make them integer
            for a in comp:
                rates[a] = rates[a] * factor

        def _makeIntegerRates(rates: Dict[str,Fraction])->Dict[str,int]:
            '''Convert integers represented as Fraction type to integer type.'''
            res = dict()
            for a in rates:
                res[a] = rates[a].numerator
            return res

        def _getAncestorCycle(tree: Dict[str,str], node1: str, node2: str):

            def _findCommonAncestor(tree: Dict[str,str], node1: str, node2: str):

                def _isAncestor(tree: Dict[str,str], node1: str, node2: str):
                    while node2 != node1 and node2 in tree:
                        node2 = tree[node2]
                    return node1 == node2

                while not _isAncestor(tree, node1, node2):
                    node1 = tree[node1]
                
                return node1

            def _ancestorPath(tree: Dict[str,str], node: str, parent: str):
                res = []
                while node != parent:
                    res.append(node)
                    node = tree[node]
                res.append(parent)
                return res                

            # start from node 1 upward and check if node is ancestor of node2, nc first one
            nc = _findCommonAncestor(tree, node1, node2)
            res = _ancestorPath(tree, node1, nc)
            res.reverse()
            res.extend(_ancestorPath(tree, node2, nc)[:-1])
            return res

        actors: List[str] = sorted(set(self._actorsAndIO))
        
        # if there are no actors return trivial solution
        if len(actors)==0:
            return dict()
        
        # keep computed fractional rates
        rates: Dict[str,Fraction] = dict()
        
        # while there are more actors to explore 
        # this loop is used once for every unconnected part of the graph
        while len(actors)>0:
            # next actor a
            a = next(iter(actors))
            
            # init to default rate
            rates[a] = Fraction(1,1)
            
            tree:Dict[str,str] = dict()
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
                if b in self._outChannels:
                    for c in self._outChannels[b]:
                        pr = self.productionRate(c)
                        co = self.consumptionRate(c)
                        ca = self._chanConsumer[c]
                        # determine the fractional rate of the connected actor ca
                        rate = rates[b] * pr / co
                        # if ca already has a rate
                        if ca in rates:
                            if not rate == rates[ca]:
                                # found inconsistent cycle
                                return _getAncestorCycle(tree, ca, b)
                        else:
                            # set b as parent of ca in the tree
                            tree[ca] = b
                            rates[ca] = rate
                            proc.append(ca)
                            comp.append(ca)
                # does b have incoming channels?
                if b in self._inChannels:
                    for c in self._inChannels[b]:
                        pr = self.productionRate(c)
                        co = self.consumptionRate(c)
                        ca = self._chanProducer[c]
                        rate = rates[b] * co / pr
                        if ca in rates:
                            if not rate == rates[ca]:
                                # found an inconsistent cycle
                                return _getAncestorCycle(tree, ca, b)
                        else:
                            tree[ca] = b
                            rates[ca] = rate
                            proc.append(ca)
                            comp.append(ca)
            _findIntegerRates(comp, rates)
        # convert fractional rates to integer rates
        return _makeIntegerRates(rates)  

    def throughput(self)->Union[Fraction,Literal['infinite']]:
        '''
        Compute throughput of the graph
        '''
        # compute state-space representation
        _, ssr = self.stateSpaceMatrices()
        # compute throughput from the state matrix
        return mpThroughput(ssr[0])

    def deadlock(self)->bool:
        '''
        Check if the dataflow graph deadlocks
        '''
        try:
            self.stateSpaceMatrices()
        except SDFDeadlockException:
            return True
        return False

    def latency(self, x0: Optional[TMPVector], mu: Fraction)->TMPMatrix:
        '''Determine the mu-periodic latency of the dataflow graph. If x0 is provided, it is considered the initial state of the initial tokens. If it is not provided, a zero vector is assumed. The latency matrix is returned. I.e., the matrix: Lambda = (C ( A-mu )^{*} ( x0 otimes [0 .inputs.. 0] oplus ( B - mu))  oplus D, where A, B, C, D are the sate-space matrices of the dataflow graph.
        '''

        _, M = self.stateSpaceMatrices()
        (A, B, C, D) = (M[0], M[1], M[2], M[3])

        if x0 is None:
            x0 = mpZeroVector(self.numberOfInitialTokens())

        # Compute the following latency matrix:
        # Lambda = (C ( A-mu )^{*} ( x0 \otimes [0 .inputs.. 0] oplus ( B - mu))  oplus D 

        Amu= mpMatrixMinusScalar(A, mu)
        try:
            scAmu = mpStarClosure(Amu)
        except PositiveCycleException:
            raise SDFException('The request period mu is smaller than smallest period the system can sustain. Therefore, it has no latency.')
        CscAmu = mpMultiplyMatrices(C, scAmu)
        x00 = mpMultiplyMatrices(mpTransposeMatrix([x0]), [mpZeroVector(len(self._inputs))])
        Bmmu= mpMatrixMinusScalar(B, mu)
        x00Bmmu = mpMaxMatrices(x00, Bmmu)
        CscAmux00Bmmu = mpMultiplyMatrices(CscAmu, x00Bmmu)
        return mpMaxMatrices(CscAmux00Bmmu, D)

    def generalizedLatency(self, mu: Fraction):
        '''Determine the mu-periodic latency of the dataflow graph in the form of separate IO-Latency and initial state latency matrices. I.e., the matrix: 
        Lambda_IO = (C ( A-mu )^{*} (B - mu)  oplus D,
        Lambda_x = (C ( A-mu )^{*}, where A, B, C, D are the sate-space matrices of the dataflow graph.
        '''

        _, M = self.stateSpaceMatrices()
        (A, B, C, D) = (M[0], M[1], M[2], M[3])

        # Lambda_IO =  = (C ( A-mu )^{*} (B - mu)  oplus D 
        # Lambda_x =   (C ( A-mu )^{*} 

        Amu= mpMatrixMinusScalar(A, mu)
        try:
            scAmu = mpStarClosure(Amu)
        except PositiveCycleException:
            raise SDFException('The request period mu is smaller than smallest period the system can sustain. Therefore, it has no latency.')
        CscAmu = mpMultiplyMatrices(C, scAmu)

        Bmmu= mpMatrixMinusScalar(B, mu)
        CscAmuBmmu = mpMultiplyMatrices(CscAmu, Bmmu)
        return CscAmu, mpMaxMatrices(CscAmuBmmu, D)

    def isSingleRate(self)->bool:
        '''Check if the graph is single-rate.'''
        for ch in self._channels:
            if ch in self._channelSpecs:
                if PROD_RATE_SPEC_KEY in self._channelSpecs[ch]:
                    if self._channelSpecs[ch][PROD_RATE_SPEC_KEY] > 1:
                        return False
                if CONS_RATE_SPEC_KEY in self._channelSpecs[ch]:
                    if self._channelSpecs[ch][CONS_RATE_SPEC_KEY] > 1:
                        return False
        return True
    
    def convertToSingleRate(self):
        '''Convert the graph to a single rate graph'''

        def _actorName(a:str, n:int, repVec: Dict[str,int])->str:
            if repVec[a] == 1:
                return a
            return '{}{}'.format(a, n+1)

        def _addChannel(res: DataflowGraph, pa: str, ca: str, it: int):
            # add channel only if it does not yet exist
            for ch in res._channels:
                if res._chanProducer[ch] == pa and res._chanConsumer[ch] == ca and res.numberOfInitialTokensOfChannel(ch) == it:
                    return
            specs = dict()
            if it > 0:
                specs[INITIAL_TOKENS_SPEC_KEY] = it
            res.addChannel(pa, ca, specs)

        # if it already is single rate, return a copy of the graph itself
        if self.isSingleRate():
            return self.copy()
        
        repVec = self.repetitionVector()
        if isinstance(repVec, list):
            raise SDFInconsistentException("Graph is inconsistent")
        
        res = DataflowGraph()

        for a in self.actorsWithoutInputsOutputs():
            if repVec[a] == 1:
                res.addActor(a, self._actorSpecs[a])
            else:
                for n in range(repVec[a]):
                    res.addActor(_actorName(a,n,repVec), self._actorSpecs[a])

        for ch in self._channels:
            it = self.numberOfInitialTokensOfChannel(ch)
            pr = self._chanProducer[ch]
            co = self._chanConsumer[ch]
            pRate = self.productionRate(ch)
            cRate = self.consumptionRate(ch)
            for n in range(repVec[pr] * pRate):
                # token n is produced by actor firing n // pRate
                pa = _actorName(pr, n//pRate, repVec)
                # token is consumed by actor firing (n+it) // cRate
                ca = _actorName(co, ((n+it) // cRate) % repVec[co], repVec)
                # number of it ((n+it) // cRate) // repVec[co]
                nit = ((n+it) // cRate) // repVec[co]
                _addChannel(res, pa, ca, nit)
       
        for i in self._inputs:
            for n in range(repVec[i]):
                res.addInputPort(_actorName(i, n, repVec))

        for o in self._outputs:
            for n in range(repVec[o]):
                res.addOutputPort(_actorName(o, n, repVec))

        for i in self._inputSignals:
            if not i in self._inputs:
                res.addInputSignal(i, self._inputSignals[i])
            else:
                seqs = mpSplitSequence(self._inputSignals[i], repVec[i])
                for n in range(repVec[i]):
                    res.addInputSignal(_actorName(i, n, repVec), seqs[n])

        return res

    def determineTrace(self, ni: int, x0: Optional[TMPVector]=None, inputOverride: Optional[Dict[str,Union[TTimeStampList,str]]]=None) -> Tuple[List[TTimeStampList],List[TTimeStampList],List[TTimeStampList],List[Fraction]]:
        '''Determine execution trace for the dataflow graph.
        The trace is ni iterations long.
        x0 is an optional initial state for the execution. If it is not provided, initial tokens are assumed to be available at time 0.
        inputOverride, is optionally used to provide input sequences to replace the ones in the model. 
        Inputs that are neither specified in the model, nor in the override, are assumed to provide all input tokens with times tamps -inf.
        Returns a tuple with the following elements
        input traces, the output traces, all firing start times, all firing durations.
        '''
        
        # determine the state-space model an the trace matrix
        H, SSM = self.stateSpaceMatrices()

        # compute vector trace from state-space matrices
        Matrices = {'A': MaxPlusMatrixModel.fromMatrix(SSM[0]), 'B': MaxPlusMatrixModel.fromMatrix(SSM[1]), 'C': MaxPlusMatrixModel.fromMatrix(SSM[2]), 'D': MaxPlusMatrixModel.fromMatrix(SSM[3]) }

        stateSize = self.numberOfInitialTokens()
        inputSize = self.numberOfInputs()

        if x0 is None:
            x0 = mpZeroVector(Matrices['A'].numberOfColumns())

        inpSig = self.inputSignals()
        inputs: List[TTimeStampList] = list()
        for s in self.inputs():
            if inputOverride and s in inputOverride:
                s: str
                if isinstance(inputOverride[s], list):
                    ios_l: TTimeStampList = inputOverride[s]  # type: ignore
                    # the input is given as an list of time stamps.
                    # split it according to the inputs within one graph iteration
                    inputs.extend(mpSplitSequence(ios_l, self.repetitions(s)))
                else:
                    # the input is given as a name referring to an input sequence specified in the model
                    ios_s: str = inputOverride[s]  # type: ignore
                    if inputOverride[s] not in inpSig:
                        raise SDFException("Unknown event sequence: {}.".format(inputOverride[s]))
                    inputs.extend(mpSplitSequence(inpSig[ios_s], self.repetitions(s)))
            else:
                # the input is not specified in override
                if s in inpSig:
                    # it is defined in the model, use it
                    inputs.extend(mpSplitSequence(inpSig[s], self.repetitions(s)))
                else:
                    # it is not specified at all, use an event sequence with minus infinity
                    inputs.extend([mpMinusInfVector(ni)] * self.repetitions(s))

        # Compute the vector trace
        vt = MaxPlusMatrixModel.vectorTrace(Matrices, x0, ni, inputs)
        
        inputTraces = [v[0:inputSize] for v in vt]
        outputTraces = [v[inputSize+stateSize:] for v in vt]

        # reorder the vectors so that the state elements come first, followed by inputs
        ssvt = [v[inputSize:stateSize+inputSize]+v[0:inputSize] for v in vt]
        # compute the firing starting times using the trace matrix H
        firingStarts = [mpMultiplyMatrixVector(H, s)  for s in ssvt]
        # collect the firing durations
        firingDurations= [self.executionTimeOfActor(a) for a in self.actorsWithoutInputsOutputs()]
        return inputTraces, outputTraces, firingStarts, firingDurations

    def determineTraceZeroBased(self, ni:int, x0: Optional[TMPVector]=None) -> Tuple[List[str],List[str],List[TTimeStampList],List[str],List[TTimeStampList],List[TTimeStampList],List[Fraction]]:
        '''Determine a trace with ni iterations, assuming that actors cannot fire before time 0.
        Optional x0 can be used to specify an initial state for the graph.
        Returns a tuple with the following elements
        actors, inputs, input traces, outputs, the output traces, all firing start times, all firing durations.
        '''
        
        # determine trace assuming actors do not start before time 0
        # clone the graph to modify it.
        G = self.copy()

        # create artificial inputs to actors to constraint their firings.
        for a in G.actorsWithoutInputsOutputs():
            inpName = '_zb_{}'.format(a)
            G.addInputPort(inpName)
            G.addChannel(inpName, a, dict())
            # set the input sequence to the new channel with tokens with time stamp 0 to prevent it from firing earlier.
            G.addInputSignal(inpName, [Fraction(0)] * ni)

        inputTraces, outputTraces, firingStarts, firingDurations = G.determineTrace(ni, x0)

        # suppress the artificial inputs
        num = len(G.actorsWithoutInputsOutputs())
        reduceRealInputs = lambda l: l[:-num]
        realInputTraces = list(map(reduceRealInputs, inputTraces))

        return G.actorsWithoutInputsOutputs(), (G.inputs())[:-num], realInputTraces, G.outputs(), outputTraces, firingStarts, firingDurations

    def asDSL(self, name: str)->str:
        '''Convert the model to a string representation in the domain specific language using the provided name.'''

        def _actorSpecs(a: str, actorsWithSpec: Set[str])->str:
            # if the specs of actor a have already been added to some instance of the actor, return an empty string
            if a in actorsWithSpec:
                return ''
            # mark that the specs have been written
            actorsWithSpec.add(a)
            # if a has no specs
            if not a in self._actorSpecs:
                return ''
            # if a has specs, but no execution time spec
            if not EXECUTION_TIME_SPEC_KEY in self._actorSpecs[a]:
                return ''
            # otherwise return the execution time spec for the DSL
            return '[{}]'.format(self._actorSpecs[a][EXECUTION_TIME_SPEC_KEY])

        def _channelSpecs(ch: str)->str:
            '''Generate the channel spec for the channel.'''
            specs = list()
            if ch in self._channelSpecs:
                if CONS_RATE_SPEC_KEY in self._channelSpecs[ch]:
                    specs.append(' consumption rate: {} '.format(self._channelSpecs[ch][CONS_RATE_SPEC_KEY]))
                if PROD_RATE_SPEC_KEY in self._channelSpecs[ch]:
                    specs.append(' production rate: {} '.format(self._channelSpecs[ch][PROD_RATE_SPEC_KEY]))
                if INITIAL_TOKENS_SPEC_KEY in self._channelSpecs[ch]:
                    specs.append(' initial tokens: {} '.format(self._channelSpecs[ch][INITIAL_TOKENS_SPEC_KEY]))                
            return ';'.join(specs)

        # create string writer for the output
        output = StringIO()
        output.write("dataflow graph {} {{\n".format(name))

        if len(self._inputs)>0:
            output.write('\tinputs ')
            output.write(', '.join(self._inputs))
            output.write('\n')

        if len(self._outputs)>0:
            output.write('\toutputs ')
            output.write(', '.join(self._outputs))
            output.write('\n')

        actorsWithSpec = set()
        for ch in self._channels:
            pr = self._chanProducer[ch]
            co = self._chanConsumer[ch]
            output.write('\t{}{} '.format(pr, _actorSpecs(pr, actorsWithSpec)))
            output.write(' ----{}----> '.format(_channelSpecs(ch)))
            output.write('{}{}\n'.format(co,  _actorSpecs(co, actorsWithSpec)))
        output.write("}\n")

        if len(self._inputSignals) > 0:
            output.write('\ninput signals\n\n')
            for inpSig in self._inputSignals:
                inputSignalRatioList = "["+", ".join(["{}".format(i) for i in self._inputSignals[inpSig]])+"]"
                output.write('{} = {}\n'.format(inpSig, inputSignalRatioList))

        result = output.getvalue()
        output.close()
        return result

    @staticmethod
    def fromDSL(dslString)->Tuple[str,'DataflowGraph']:
        '''
        Parse the model provides as a string.
        Returns if successful a pair with the name of the model and the constructed instance of `DataflowGraph`
        '''

        factory = dict()
        factory['Init'] = lambda : DataflowGraph()
        factory['AddActor'] = lambda sdf, a, specs: sdf.addActor(a, specs)
        factory['AddChannel'] = lambda sdf, a1, a2, specs: sdf.addChannel(a1, a2, specs)
        factory['AddInputPort'] = lambda sdf, i: sdf.addInputPort(i)
        factory['AddOutputPort'] = lambda sdf, i: sdf.addOutputPort(i)
        factory['AddInputSignal'] = lambda sdf, n, s: sdf.addInputSignal(n, s)
        result = parseSDFDSL(dslString, factory)
        if result[0] is None:
            exit(1)
        return result  # type: ignore


    def __str__(self):
        return "({}, {}, {}, {})".format(self._actorsAndIO, self._channels, self._inputs, self._outputs)
