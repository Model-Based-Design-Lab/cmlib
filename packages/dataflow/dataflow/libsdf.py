import copy
from functools import reduce
from io import StringIO
from typing import Dict, List, Optional
from dataflow.libsdfgrammar import parseSDFDSL
from dataflow.maxplus.starclosure import PositiveCycleException
from dataflow.maxplus.maxplus import mpThroughput, mpMatrixMinusScalar, mpStarClosure, mpMultiplyMatrices, mpMultiplyMatrixVector, mpMinusInfVector, mpTransposeMatrix, mpZeroVector, mpMaxMatrices, mpMaxVectors, mpScaleVector, mpSplitSequence, TTimeStamp
from dataflow.libmpm import MaxPlusMatrixModel
from fractions import Fraction

# constants
DEFAULT_ACTOR_EXECUTION_TIME = 1.0
EXECUTION_TIME_SPEC_KEY = 'executionTime'
CONS_RATE_SPEC_KEY = 'consRate'
PROD_RATE_SPEC_KEY = 'prodRate'
INITIAL_TOKENS_SPEC_KEY = 'initialTokens'

def _splitMatrix(M, n):
    A = []
    B = []
    for k in range(n):
        A.append(M[k][:n])
        B.append(M[k][n:])
    C = []
    D = []
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
 
    def __init__(self):
        # set of actors, including inputs and outputs!
        self._actors = list()
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

    def copy(self):
        ''' Return a new DataflowGraph as a copy of thw one the method of which
        is called/ '''
        return copy.deepcopy(self)
        

    def actors(self):
        return self._actors

    def channels(self):
        return self._channels

    def actorsWithoutInputsOutputs(self):
        return [a for a in self._actors if not (a in self._inputs or a in self._outputs)]

    def inputs(self):
        return self._inputs

    def outputs(self):
        return self._outputs

    def inputSignals(self):
        return self._inputSignals

    def consumptionRate(self, ch):
        if ch in self._channelSpecs:
            if CONS_RATE_SPEC_KEY in self._channelSpecs[ch]:
                return self._channelSpecs[ch][CONS_RATE_SPEC_KEY]
        return 1

    def productionRate(self, ch):
        if ch in self._channelSpecs:
            if PROD_RATE_SPEC_KEY in self._channelSpecs[ch]:
                return self._channelSpecs[ch][PROD_RATE_SPEC_KEY]
        return 1

    def repetitions(self, actor):
        '''
        Determine repetition vector entry for actor
        '''
        if self._repetitionVector is None:
            self._repetitionVector = self.repetitionVector()
            if self._repetitionVector is None:
                raise SDFInconsistentException("Dataflow graph is inconsistent")
        return self._repetitionVector[actor]

    def repetitionVectorSum(self):
        if self._repetitionVector is None:
            self._repetitionVector = self.repetitionVector()
            if self._repetitionVector is None:
                raise SDFInconsistentException("Dataflow graph is inconsistent")
        res = 0
        for a in self._actors:
            res = res + self._repetitionVector[a]
        return res

    def validate(self):
        '''
        Validate the dataflow model. Raises an exception on an invalid model. Returns nothing.
        '''
        unreadInputs = set(self._inputs).difference(set(self._actors))
        if len(unreadInputs) > 0:
            raise SDFException('Invalid model. The following inputs are not read: {}.'.format('. '.join(unreadInputs)))
        unwrittenOutputs = set(self._outputs).difference(set(self._actors))
        if len(unwrittenOutputs) > 0:
            raise SDFException('Invalid model. The following outputs are not written: {}.'.format('. '.join(unwrittenOutputs)))

    def addActor(self, a, specs):
        # invalidate cached repetition vector
        self._repetitionVector = None
        # add actor if it doesn't exist
        if not a in self._actors:
            self._actors.append(a)
            self._actorSpecs[a] = dict()
        # add specs
        for s in specs:
            self._actorSpecs[a][s] = specs[s]

    def addChannel(self, a, b, specs):
        self._repetitionVector = None
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

    def addInputPort(self, i):
        self._repetitionVector = None
        # input ports should be in actors
        if not i in self._actors:
            self.addActor(i, {EXECUTION_TIME_SPEC_KEY: 0.0})
        else:
            self._actorSpecs[i][EXECUTION_TIME_SPEC_KEY] = 0.0
        self._inputs.append(i)

    def addOutputPort(self, o):
        self._repetitionVector = None
        # output ports should be in actors
        if not o in self._actors:
            self.addActor(o, {EXECUTION_TIME_SPEC_KEY: 0.0})
        else:
            self._actorSpecs[o][EXECUTION_TIME_SPEC_KEY] = 0.0
        self._outputs.append(o)

    def addInputSignal(self, n, s):
        self._repetitionVector = None
        self._inputSignals[n] = s

    def producerOfChannel(self, ch):
        return self._chanProducer[ch]

    def consumerOfChannel(self, ch):
        return self._chanConsumer[ch]

    def _newChannelName(self):
        fname = lambda m: 'ch'+str(m)
        k = 1
        while fname(k) in self._channelSpecs:
            k += 1
        return fname(k)

    def executionTimeOfActor(self, a):
        if not EXECUTION_TIME_SPEC_KEY in self._actorSpecs[a]:
            return DEFAULT_ACTOR_EXECUTION_TIME
        return self._actorSpecs[a][EXECUTION_TIME_SPEC_KEY]

    def specsOfActor(self, a):
        return self._actorSpecs[a]

    def numberOfInitialTokensOfChannel(self, ch):
        if not INITIAL_TOKENS_SPEC_KEY in self._channelSpecs[ch]:
            return 0
        return self._channelSpecs[ch][INITIAL_TOKENS_SPEC_KEY]

    def setNumberOfInitialTokensOfChannel(self, ch, it):
        self._channelSpecs[ch][INITIAL_TOKENS_SPEC_KEY] = it

    def numberOfInitialTokens(self):
        return reduce(lambda sum, ch: sum + self.numberOfInitialTokensOfChannel(ch), self._channels, 0)

    def numberOfInputs(self):
        return len(self._inputs)

    def numberOfInputsInIteration(self):
        res = 0
        for i in self._inputs:
            res = res + self.repetitions(i)
        return res
    
    def channelSet(self):
        result = set()
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

    def inChannels(self, a):
        if a in self._inChannels:
            return self._inChannels[a]
        return set()

    def outChannels(self, a):
        if a in self._outChannels:
            return self._outChannels[a]
        return set()

    def listOfInputsStr(self):
        return ', '.join(self._inputs)

    def listOfOutputsStr(self):
        return ', '.join(self._outputs)

    def stateElementLabels(self):
        res = []
        for ch in self._channels:
            for n in range(self.numberOfInitialTokensOfChannel(ch)):
                res.append(self._readableInitialTokenLabel(ch, n))
        return res

    def listOfStateElementsStr(self):
        return ', '.join(self.stateElementLabels())


    def _initialTokenLabel(self, ch, n):
        if self.numberOfInitialTokensOfChannel(ch) == 1:
            return ch
        return ch+'_'+str(n)

    def _inputTokenLabel(self, i, n):
        if self.repetitions(i) == 1:
            return i
        return i+'_'+str(n)

    def _outputTokenLabel(self, o, n):
        if self.repetitions(o) == 1:
            return o
        return o+'_'+str(n)

    def _actorFiringLabel(self, a, n):
        if self.repetitions(a) == 1:
            return a
        return a+'_'+str(n)


    def _readableInitialTokenLabel(self, ch, n):
        if self.numberOfInitialTokensOfChannel(ch) < 2:
            return '{}_{}'.format(self._chanProducer[ch], self. _chanConsumer[ch])
        else:
            return '{}_{}_{}'.format(self._chanProducer[ch], self. _chanConsumer[ch], n)

    def _initializeSymbolicTimeStamps(self):
        self._symbolicTimeStampSize = self.numberOfInitialTokens() + self.numberOfInputsInIteration()
        self._symbolicVector = []
        for ch in self._channels:
            for n in range(self.numberOfInitialTokensOfChannel(ch)):
                self._symbolicVector.append(self._initialTokenLabel(ch, n))
        for i in self._inputs:
            for n in range(self.repetitions(i)):
                self._symbolicVector.append(self._inputTokenLabel(i, n))

    def _symbolicTimeStampMinusInfinity(self) -> List[TTimeStamp]:
        return [None] * self._symbolicTimeStampSize

    def symbolicTimeStamp(self, t):
        res = self._symbolicTimeStampMinusInfinity()
        res[self._symbolicVector.index(t)] = 0
        return res

    def _symbolicTimeStampMax(self, ts1, ts2):
        return mpMaxVectors(ts1, ts2)

    def _symbolicTimeStampScale(self, c, ts):
        return mpScaleVector(c, ts)

    def _symbolicFiring(self, a, n, timestamps):
        el = self._actorFiringLabel(a, n)
        if el in timestamps:
            # I already have a time stamp
            return False
        # check if my dependencies are complete
        ts = self._symbolicTimeStampMinusInfinity()
        for ch in self.inChannels(a):
            cr = self.consumptionRate(ch)
            cons = cr * (n+1)
            if self.numberOfInitialTokensOfChannel(ch) >= cons:
                it = cons
            else:
                it = self.numberOfInitialTokensOfChannel(ch)

            for k in range(it):
                ts = self._symbolicTimeStampMax(ts, timestamps[self._initialTokenLabel(ch, k)])

            rem = cons - it
            b = self._chanProducer[ch]
            pr = self.productionRate(ch)
            for k in range(rem):
                # which firing produced token n ?
                m = k // pr
                elm = self._actorFiringLabel(b, m)
                if not elm in timestamps:
                    return False
                ts = self._symbolicTimeStampMax(ts, self._symbolicTimeStampScale(self.executionTimeOfActor(b), timestamps[elm]))
        timestamps[el] = ts
        return True


    def _symbolicCompletionTime(self, timestamps, a, n):
        el = self._actorFiringLabel(a, n)
        return self._symbolicTimeStampScale(self.executionTimeOfActor(a), timestamps[el])

    def stateSpaceMatrices(self):
        '''
        Compute the trace matrix and the state-space, A, B, C, and D, matrices.
        Returns a pair with 
        - the trace matrix (H) with a row for every actor firing in an iteration
        - a four-tuple with the state-space matrices, A, B, C and D
        '''
        self._initializeSymbolicTimeStamps() 
        timestamps = dict()
        for i in self._inputs:
            for n in range(self.repetitions(i)):
                el = self._inputTokenLabel(i, n)
                timestamps[el] = self.symbolicTimeStamp(el)
        for ch in self._channels:
            for n in range(self.numberOfInitialTokensOfChannel(ch)):
                el = self._initialTokenLabel(ch, n)
                timestamps[el] = self.symbolicTimeStamp(el)
        actorFirings = dict()
        for a in self._actors:
            actorFirings[a] = 0
        oldLen = len(timestamps)
        while len(timestamps)<self.repetitionVectorSum()+self.numberOfInitialTokens():
            for a in self._actors:
                if actorFirings[a] < self.repetitions(a):
                    el = self._actorFiringLabel(a, actorFirings[a])
                    if not el in timestamps:
                        if self._symbolicFiring(a, actorFirings[a], timestamps):
                            actorFirings[a] += 1
            if len(timestamps) == oldLen:
                # print(timestamps)
                raise SDFDeadlockException("The graph deadlocks.")
            oldLen = len(timestamps)
        A = []
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

        for o in self._outputs:
            for n in range(self.repetitions(o)):
                el = self._outputTokenLabel(o, n)
                A.append(timestamps[el])

        H = []
        for a in self._actors:
            if not (a in self._inputs or a in self._outputs):
                for n in range(self.repetitions(a)):
                    el = self._actorFiringLabel(a, n)
                    H.append(timestamps[el])

        return H, _splitMatrix(A, self.numberOfInitialTokens())

    def repetitionVector(self) -> Optional[Dict[str,int]]:

        def _findIntegerRates(comp, rates):
            factor = Fraction(1,1)
            for a in comp:
                scRate = factor*rates[a]
                if scRate.denominator > 1:
                    factor = factor * scRate.denominator
            for a in comp:
                rates[a] = rates[a] * factor

        def _makeIntegerRates(rates):
            res = dict()
            for a in rates:
                res[a] = rates[a].numerator
            return res

        actors = set(self._actors)
        if len(actors)==0:
            return dict()
        rates = dict()
        while len(actors)>0:
            a = next(iter(actors))
            rates[a] = Fraction(1,1)
            tree = dict()
            proc = set([a])
            comp = set([a])
            while len(proc)>0:
                b = next(iter(proc))
                proc.remove(b)
                actors.remove(b)
                if b in self._outChannels:
                    for c in self._outChannels[b]:
                        pr = self.productionRate(c)
                        co = self.consumptionRate(c)
                        ca = self._chanConsumer[c]
                        rate = rates[b] * pr / co
                        if ca in rates:
                            if not rate == rates[ca]:
                                # find inconsistent cycle
                                return None
                        else:
                            tree[ca] = b
                            rates[ca] = rate
                            proc.add(ca)
                            comp.add(ca)
                if b in self._inChannels:
                    for c in self._inChannels[b]:
                        pr = self.productionRate(c)
                        co = self.consumptionRate(c)
                        ca = self._chanProducer[c]
                        rate = rates[b] * co / pr
                        if ca in rates:
                            if not rate == rates[ca]:
                                # find inconsistent cycle
                                return None
                        else:
                            tree[ca] = b
                            rates[ca] = rate
                            proc.add(ca)
                            comp.add(ca)
            _findIntegerRates(comp, rates)

        return _makeIntegerRates(rates)  

    def throughput(self):
        '''
        Compute throughput of the graph
        '''
        # compute state-space representation
        _, ssr = self.stateSpaceMatrices()
        # compute throughput from the state matrix
        return mpThroughput(ssr[0])

    def deadlock(self):
        '''
        Check if the dataflow graph deadlocks
        '''
        try:
            self.stateSpaceMatrices()
        except SDFDeadlockException:
            return True
        return False

    def latency(self, x0, mu):

        _, M = self.stateSpaceMatrices()
        (A, B, C, D) = (M[0], M[1], M[2], M[3])

        if x0 is None:
            x0 = [0] * self.numberOfInitialTokens()

        # Lmbd = (C ( A-mu )^{*} ( x0 \otimes [0 .inputs.. 0] oplus ( B - mu))  oplus D 

        Amu= mpMatrixMinusScalar(A, mu)
        try:
            scAmu = mpStarClosure(Amu)
        except PositiveCycleException:
            raise SDFException('The period is smaller than smallest period the system can sustain.')
        CscAmu = mpMultiplyMatrices(C, scAmu)
        x00 = mpMultiplyMatrices(mpTransposeMatrix([x0]), [mpZeroVector(len(self._inputs))])
        Bmmu= mpMatrixMinusScalar(B, mu)
        x00Bmmu = mpMaxMatrices(x00, Bmmu)
        CscAmux00Bmmu = mpMultiplyMatrices(CscAmu, x00Bmmu)
        return mpMaxMatrices(CscAmux00Bmmu, D)

    def generalizedLatency(self, mu):

        _, M = self.stateSpaceMatrices()
        (A, B, C, D) = (M[0], M[1], M[2], M[3])

        # Lmbd_IO =  = (C ( A-mu )^{*} (B - mu)  oplus D 
        # Lmbd_x =   (C ( A-mu )^{*} 

        Amu= mpMatrixMinusScalar(A, mu)
        try:
            scAmu = mpStarClosure(Amu)
        except PositiveCycleException:
            raise SDFException('The period is smaller than smallest period the system can sustain.')
        CscAmu = mpMultiplyMatrices(C, scAmu)

        Bmmu= mpMatrixMinusScalar(B, mu)
        CscAmuBmmu = mpMultiplyMatrices(CscAmu, Bmmu)
        return CscAmu, mpMaxMatrices(CscAmuBmmu, D)

    def isSingleRate(self):
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

        def _actorName(a, n, repVec):
            if repVec[a] == 1:
                return a
            return '{}{}'.format(a, n+1)

        def _addChannel(res, pa, ca, it):
            # add channel only if it does not yet exist
            for ch in res._channels:
                if res._chanProducer[ch] == pa and res._chanConsumer[ch] == ca and res.numberOfInitialTokensOfChannel(ch) == it:
                    return
            specs = dict()
            if it > 0:
                specs[INITIAL_TOKENS_SPEC_KEY] = it
            res.addChannel(pa, ca, specs)


        if self.isSingleRate():
            return self
        
        repVec = self.repetitionVector()
        if repVec is None:
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

    def determineTrace(self, ni, x0=None, inputOverride=None):
        H, SSM = self.stateSpaceMatrices()

        # compute vector trace from state-space matrices
        Matrices = {'A': MaxPlusMatrixModel.fromMatrix(SSM[0]), 'B': MaxPlusMatrixModel.fromMatrix(SSM[1]), 'C': MaxPlusMatrixModel.fromMatrix(SSM[2]), 'D': MaxPlusMatrixModel.fromMatrix(SSM[3]) }

        stateSize = self.numberOfInitialTokens()
        inputSize = self.numberOfInputs()

        if x0 is None:
            x0 = mpZeroVector(Matrices['A'].numberOfColumns())

        inpSig = self.inputSignals()
        inputs = list()
        for s in self.inputs():
            if inputOverride and s in inputOverride:
                if isinstance(inputOverride[s], list):
                    inputs.extend(mpSplitSequence(inputOverride[s], self.repetitions(s)))
                else:
                    if inputOverride[s] not in inpSig:
                        raise SDFException("Unknown event sequence: {}.".format(inputOverride[s]))
                    inputs.extend(mpSplitSequence(inpSig[inputOverride[s]], self.repetitions(s)))
            else:
                if s in inpSig:
                    inputs.extend(mpSplitSequence(inpSig[s], self.repetitions(s)))
                else:
                    inputs.extend([mpMinusInfVector(ni)] * self.repetitions(s))

        vt = MaxPlusMatrixModel.vectorTrace(Matrices, x0, ni, inputs)
        ssvt = [v[inputSize:stateSize+inputSize]+v[0:inputSize] for v in vt]
        
        inputTraces = [v[0:inputSize] for v in vt]
        outputTraces = [v[inputSize+stateSize:] for v in vt]
        firingStarts = [mpMultiplyMatrixVector(H, s)  for s in ssvt]
        firingDurations= [self.executionTimeOfActor(a) for a in self.actorsWithoutInputsOutputs()]
        return inputTraces, outputTraces, firingStarts, firingDurations

    def determineTraceZeroBased(self, ni, x0=None):
        # determine trace assuming actors do not start before time 0

        G = self.copy()

        for a in G.actorsWithoutInputsOutputs():
            inpName = '_zb_{}'.format(a)
            G.addInputPort(inpName)
            G.addChannel(inpName, a, dict())
            G.addInputSignal(inpName, [0] * ni)

        inputTraces, outputTraces, firingStarts, firingDurations = G.determineTrace(ni, x0)

        # suppress the artificial inputs
        num = len(G.actorsWithoutInputsOutputs())
        reduceRealInputs = lambda l: l[:-num]
        realInputTraces = list(map(reduceRealInputs, inputTraces))

        return G.actorsWithoutInputsOutputs(), (G.inputs())[:-num], realInputTraces, G.outputs(), outputTraces, firingStarts, firingDurations

    def asDSL(self, name):

        def _actorSpecs(a, actorsWithSpec):
            if a in actorsWithSpec:
                return ''
            actorsWithSpec.add(a)
            if not a in self._actorSpecs:
                return ''
            if not EXECUTION_TIME_SPEC_KEY in self._actorSpecs[a]:
                return ''
            return '[{}]'.format(self._actorSpecs[a][EXECUTION_TIME_SPEC_KEY])

        def _channelSpecs(ch):
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
            for inpsig in self._inputSignals:
                output.write('{} = {}\n'.format(inpsig, self._inputSignals[inpsig]))

        result = output.getvalue()
        output.close()
        return result

    @staticmethod
    def fromDSL(dslString):
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
        return result


    def __str__(self):
        return "({}, {}, {}, {})".format(self._actors, self._channels, self._inputs, self._outputs)
