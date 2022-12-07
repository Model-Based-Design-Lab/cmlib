import math
from statistics import NormalDist
from typing import List, Optional, Tuple, Union

# As a rule of thumb, a reasonable number of results are needed before certain calculations can be considered valid
# this variable determines the number of results that are required for the markov simulation before the stop conditions
# are checked. (Based on the law of strong numbers)
_law: int = 30


RES_TOO_FEW_SAMPLES = "Cannot be decided, too few samples."
class Statistics(object):
    '''
    Determine estimated long-run sample average, absolute error, relative error and confidence interval
    Equations follow Section B.7.3. of the Reader 5xie0, Computational Modeling
    '''

    _cycleLength:int                            # Current cycle length Ln
    _sumOfSamplesCycle: float                   # Current cycle, sum of samples
    _cycleCount: int                            # Cycle count 
    _cumulativeCycleLengths: int                # Sum of cycle lengths
    _cumulativeSamples: float                   # Cumulative sum of samples
    _cumulativeCycleLengthsSq: int              # Sum of cycle length squared
    _cumulativeSamplesCycleSq: float            # Sum of cycle cumulative samples squared
    _cumulativeProdCycleLengthSumSamples: float # Sum of cycle product length and cycle
    _meanEst: float                                # Estimated mean
    _stdDevEst: float                            # Estimated variance
    _confLevel: float                           # confidence level

    def __init__(self, confidence: float) -> None:
        '''confidence: confidence level'''
        self._cycleLength = 0                           # Current cycle length
        self._sumOfSamplesCycle = 0.0                   # Current cycle cumulative samples
        self._cycleCount = 0                           # Cycle count (-1 to subtract unfinished cycle beforehand)
        self._cumulativeCycleLengths = 0                # Sum of cycle lengths
        self._cumulativeSamples = 0.0                   # Sum of cumulative samples
        self._cumulativeCycleLengthsSq = 0              # Sum of cycle length squared
        self._cumulativeSamplesCycleSq = 0              # Sum of cycle cumulative samples squared
        self._cumulativeProdCycleLengthSumSamples = 0   # Sum of cycle product length and cycle
        self._meanEst = 0                                  # Estimated mean
        self._stdDevEst = 0                              # Estimated variance

        # Calculate the confidence point estimate with inverse normal distribution
        self._confLevel = NormalDist().inv_cdf((1+confidence)/2)

    def updateMeanEstimate(self)->None:
        # Update point estimate of mean, Eq. B.66.
        if self._cumulativeCycleLengths != 0:
            self._meanEst = self._cumulativeSamples/self._cumulativeCycleLengths

    def meanEstimate(self)->Optional[float]:
        if self._cycleCount < _law:
            return None
        return self._meanEst

    def meanEstimateResult(self)->Union[str,float]:
        if self._cycleCount < _law:
            return RES_TOO_FEW_SAMPLES
        return self._meanEst

    def stdDevEstimate(self)->Optional[float]:
        if self._cycleCount < _law:
            return None
        return self._stdDevEst

    def updateVarianceEstimate(self)->None:
        # Compute S_M following the equation below Eq. B.67 of the reader
        # Update point estimate of standard deviation.
        # assumes mean estimate is up to date!
        if (self._cumulativeCycleLengths != 0) and (self._cycleCount != 0):
            self._stdDevEst = math.sqrt(abs(
                self._cumulativeSamplesCycleSq 
                - 2*self._meanEst*self._cumulativeProdCycleLengthSumSamples 
                + pow(self._meanEst,2)*self._cumulativeCycleLengthsSq
            )/self._cycleCount)

    def completeCycle(self)->None:
        self._cycleCount += 1
        self._cumulativeCycleLengths += self._cycleLength
        self._cumulativeSamples += self._sumOfSamplesCycle
        self._cumulativeCycleLengthsSq += pow(self._cycleLength, 2)
        self._cumulativeSamplesCycleSq += pow(self._sumOfSamplesCycle, 2)
        self._cumulativeProdCycleLengthSumSamples += self._cycleLength * self._sumOfSamplesCycle
        
        # reset accumulators
        self._cycleLength = 0
        self._sumOfSamplesCycle = 0.0

        self.updateMeanEstimate()
        self.updateVarianceEstimate()

    def addSample(self, s: float)->None:
        self._cycleLength += 1
        self._sumOfSamplesCycle += s

    def cycleCount(self)->int:
        return self._cycleCount

    def abError(self)->Optional[float]:
        '''
        Return estimated absolute error.
        If no estimate can be given, None is returned
        '''

        # check if we have collected sufficient cycles 
        if self._cycleCount < _law:
            return None
        
        if self._cycleCount == 0:
            return None
        
        # denominator of the absolute error term in Eq. B.69
        den = self._cumulativeCycleLengths/math.sqrt(self._cycleCount)
        if den != 0.0:
            return abs((self._confLevel*self._stdDevEst) / den)
        else:
            return None

    def reError(self)->Optional[float]:
        '''Return estimated relative error'''

        # check if we have collected sufficient cycles 
        if 0 <= self._cycleCount < _law:
            return None

        if self._cycleCount == 0:
            return None

        absError = self.abError()
        if absError is None:
            return None

        if absError >= abs(self._meanEst):
            return None

        return absError / (abs(self._meanEst) - absError)

    def confidenceInterval(self)->Optional[Tuple[float,float]]:
        # Compute confidence interval
        abError = self.abError()
        if abError is None:
            return None
        return (self._meanEst - abError, self._meanEst + abError)


class DistributionStatistics(object):
    '''Determine Cesaro limit distribution'''

    _number_of_states: int  # length of distribution 
    _stateEstimators: List[Statistics]

    def __init__(self, nr_of_states: int, confidence: float) -> None:
        self._number_of_states = nr_of_states
        self._stateEstimators = [Statistics(confidence) for i in range(nr_of_states)]

    def completeCycle(self)->None:
        for s in self._stateEstimators:
            s.completeCycle()

    def visitState(self, n: int)->None:
        for i in range(self._number_of_states):
            if i == n:
                self._stateEstimators[i].addSample(1.0)
            else:
                self._stateEstimators[i].addSample(0.0)

    def cycleCount(self)->int:
        return self._stateEstimators[0].cycleCount()

    def pointEstimates(self)->Optional[List[float]]:
        res = [s.meanEstimate() for s in self._stateEstimators]
        if None in res:
            return None
        vRes: List[float] = res  # type: ignore
        if not 0.9999 < sum(vRes) < 1.0001:
            return None
        
        return vRes

    def stdDevEstimates(self)->Optional[List[float]]:
        res = [s.stdDevEstimate() for s in self._stateEstimators]
        if None in res:
            return None
        vRes: List[float] = res  # type: ignore        
        return vRes

    def abError(self)->List[Optional[float]]:
        '''Return estimated absolute errors'''
        return [s.abError() for s in self._stateEstimators]

    def maxAbError(self)->Optional[float]:
        '''Return maximum estimated absolute error'''
        res = [s.abError() for s in self._stateEstimators]
        if any([e is None for e in res]):
            return None
        vRes: List[float] = res  # type: ignore
        return max(vRes)

    def reError(self)->List[Optional[float]]:
        '''Return estimated relative errors'''
        return [s.reError() for s in self._stateEstimators]

    def maxReError(self)->Optional[float]:
        '''Return maximum estimated relative error'''
        res = [s.reError() for s in self._stateEstimators]
        if any([e is None for e in res]):
            return None
        vRes: List[float] = res  # type: ignore
        return max(vRes)

    def confidenceIntervals(self)->Optional[List[Tuple[float,float]]]:
        res = [s.confidenceInterval() for s in self._stateEstimators]
        if any([i is None for i in res]):
            return None
        vRes: List[Tuple[float,float]] = res  # type: ignore
        return vRes
