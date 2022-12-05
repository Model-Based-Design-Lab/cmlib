import math
from statistics import NormalDist
from typing import List, Optional, Tuple
import markovchains.utils.linalgebra as linalg

# As a rule of thumb, a reasonable number of results are needed before certain calculations can be considered valid
# this variable determines the number of results that are required for the markov simulation before the stop conditions
# are checked. (Based on the law of strong numbers)
_law: int = 30


# TODO: modify terminology: reward to sample ?
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
    _cumulativeSamplesCycleSq: float            # Sum of cycle cumulative reward squared
    _cumulativeProdCycleLengthSumSamples: float # Sum of cycle product length and cycle
    _meanEst: float                                # Estimated mean
    _stdDevEst: float                            # Estimated variance
    _confLevel: float                           # confidence level

    def __init__(self, confidence: float) -> None:
        '''confidence: confidence level'''
        self._cycleLength = 0                           # Current cycle length
        self._sumOfSamplesCycle = 0.0                   # Current cycle cumulative reward
        self._cycleCount = 0                           # Cycle count (-1 to subtract unfinished cycle beforehand)
        self._cumulativeCycleLengths = 0                # Sum of cycle lengths
        self._cumulativeSamples = 0.0                   # Sum of cumulative rewards
        self._cumulativeCycleLengthsSq = 0              # Sum of cycle length squared
        self._cumulativeSamplesCycleSq = 0              # Sum of cycle cumulative reward squared
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

    def abError(self, noWarmup: bool = False)->Optional[float]:
        '''
        Return estimated absolute error.
        If no estimate can be given, None is returned
        If noWarmup = True is provided, then an estimate is returned, even if the minimum number of cycles is not yet achieved.
        '''

        # TODO: check if noWarmup is actually used

        # check if we have collected sufficient cycles 
        if not noWarmup and (self._cycleCount < _law):
            return None
        
        if self._cycleCount == 0:
            return None
        
        # denominator of the absolute error term in Eq. B.69
        den = self._cumulativeCycleLengths/math.sqrt(self._cycleCount)
        if den != 0.0:
            return abs((self._confLevel*self._stdDevEst) / den)
        else:
            return None

    def reError(self, noWarmup: bool = False)->Optional[float]:
        '''Return estimated relative error'''

        # check if we have collected sufficient cycles 
        if not noWarmup and (0 <= self._cycleCount < _law):
            return None

        if self._cycleCount == 0:
            return None

        absError = self.abError(noWarmup)
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

    def abError(self, noWarmup: bool = False)->List[Optional[float]]:
        '''Return estimated absolute errors'''
        return [s.abError(noWarmup) for s in self._stateEstimators]

    def reError(self, noWarmup: bool = False)->List[Optional[float]]:
        '''Return estimated relative errors'''
        return [s.reError(noWarmup) for s in self._stateEstimators]

    def confidenceIntervals(self)->List[Optional[Tuple[float,float]]]:
        return [s.confidenceInterval() for s in self._stateEstimators]


# what is this for?

# class WelfordWiki(object):
#     '''
#     https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
#     '''

#     def __init__(self) -> None:
#         self._count = 0
#         self._mean = 0.0
#         self._M2 = 0.0

#     def update(self, newValue):
#         # For a new value newValue, compute the new count, new mean, the new M2.
#         # mean accumulates the mean of the entire dataset
#         # M2 aggregates the squared distance from the mean
#         # count aggregates the number of samples seen so far
#         self._count += 1
#         delta = newValue - self._mean
#         self._mean += delta / self._count
#         delta2 = newValue - self._mean
#         self._M2 += delta * delta2

#     def finalize(self, existingAggregate):
#         # Retrieve the mean, variance and sample variance from an aggregate
#         if self._count < 2:
#             return float("nan")
#         else:
#             (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
#             return (mean, variance, sampleVariance)

# class Welford(object):

#     def __init__(self, confidence: float, stopDescriptions: List[str]) -> None:

#         # confidence level
#         self._c = NormalDist().inv_cdf((1+confidence)/2)

#         self._interval: Tuple[float,float] = (0, 0)
#         self._abErrorVal: float = -1.0
#         self._reErrorVal: float = -1.0
#         self._mean: float = 0.0
#         self._count: int = 0
#         self._M2: float = 0.0 # Welford's algorithm variable

#         # There are in total four applicable stop conditions for this function
#         self._y = 0.0

#     def sample(self, v: float)->None:
#         self._y = v
#         # Execute Welford's algorithm to compute running standard derivation and mean
#         self._count += 1
#         d1 = self._y - self._mean
#         self._mean += d1/self._count
#         d2 = self._y - self._mean
#         self._M2 += d1 * d2
#         self._Sm = math.sqrt(self._M2/float(self._count))

#     def getNumberOfSamples(self)->int:
#         return self._count

#     def getErrorVals(self)->Tuple[float,float]:

#         # Compute absolute and relative errors
#         self._abErrorVal = abs((self._c*self._Sm) / math.sqrt(float(self._count)))
#         self._d = self._mean-self._abErrorVal
#         if self._d != 0.0:
#             self._reErrorVal = abs(self._abErrorVal/self._d)

#         # interval calculation
#         self._interval = (self._mean - self._abErrorVal, self._mean + self._abErrorVal)

#         # Do not evaluate abError/reError in the first _law cycles:
#         if self._count < _law and self._count != rounds: # (if rounds is less than 10 we still want an abError and reError)
#             self._abErrorVal = -1.0
#             self._reErrorVal = -1.0
#         return self._abError, self._reError