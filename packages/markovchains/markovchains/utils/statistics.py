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
    '''Determine estimated long-run-average cumulative reward'''

    _l:int      # Current cycle length
    _r: float   # Current cycle cumulative reward
    _Em: int    # Cycle count (-1 to subtract unfinished cycle beforehand)
    _El: int    # Sum of cycle lengths
    _Er: float  # Sum of cumulative rewards
    _El2: int   # Sum of cycle length squared
    _Er2: float # Sum of cycle cumulative reward squared
    _Elr: float # Sum of cycle product length and cycle
    _u: float   # Estimated mean
    _Sm: float  # Estimated variance
    _con: float # confidence level

    def __init__(self, confidence: float) -> None:
        '''confidence: confidence level'''
        self._l = 0    # Current cycle length
        self._r = 0.0  # Current cycle cumulative reward
        self._Em = -1  # Cycle count (-1 to subtract unfinished cycle beforehand)
        self._El = 0   # Sum of cycle lengths
        self._Er = 0.0 # Sum of cumulative rewards
        self._El2 = 0  # Sum of cycle length squared
        self._Er2 = 0  # Sum of cycle cumulative reward squared
        self._Elr = 0  # Sum of cycle product length and cycle
        self._u = 0    # Estimated mean
        self._Sm = 0   # Estimated variance

        # Calculate the confidence point estimate with inverse normal distribution
        self._con = NormalDist().inv_cdf((1+confidence)/2)


    def visitRecurrentState(self)->None:
        self._Em += 1
        self._El += self._l
        self._Er += self._r
        self._El2 += pow(self._l, 2)
        self._Er2 += pow(self._r, 2)
        self._Elr += self._l * self._r
        self._l = 0
        self._r = 0.0

        # Update point estimate of mean.
        if self._El != 0:
            self._u = self._Er/self._El

        # Update point estimate of variance.
        if (self._El != 0) and (self._Em != 0):
            self._Sm = math.sqrt(abs((self._Er2 - 2*self._u*self._Elr + pow(self._u,2)*self._El2)/self._Em))


    def addReward(self, r: float)->None:
        self._l += 1
        self._r += r

    def cycleCount(self)->int:
        return self._Em

    def abError(self, noWarmup: bool = False)->Optional[float]:
        '''Return estimated absolute error'''

        # Run first _law times without checking abError
        if not noWarmup and (0 <= self._Em < _law):
            return None
        if self._Em > 0:
            d = math.sqrt(self._Em) * (1/(self._Em)) * self._El
            if d != 0:
                return abs((self._con*self._Sm) / d)
            else:
                return None
        else:
            return None

    def reError(self, noWarmup: bool = False)->Optional[float]:
        '''Return estimated relative error'''
        # Run first 10 times without checking abError
        if not noWarmup and (0 <= self._Em < _law):
            return None

        if self._Em > 0:
            d = (self._u * math.sqrt(self._Em) * (1/(self._Em)) * self._El) - (self._con*self._Sm)
            if d != 0:
                return abs((self._con*self._Sm) / d)
            else:
                return None
        else:
            return None

    def sanitizedRelativeError(self)->Optional[float]:
        '''Return estimated relative error'''
        if 0 <= self._Em < _law:
            return None

        reError = self.reError(noWarmup=True)
        if invalid:
            return None
        return reError

    def confidenceInterval(self)->Tuple[float,float]:
        # Compute confidence interval
        abError = self.abError(noWarmup=True)
        return (self._u - abError, self._u + abError)


class DistributionStatistics(object):
    '''Determine Cesaro limit distribution'''

    _l: int                 # Current cycle length
    _Em: int                # Cycle count (-1 to subtract unfinished cycle beforehand)
    _El: int                # Sum of cycle lengths
    _El2: int               # Sum of cycle length squared
    _rl: List[float]        # Current cycle cumulative reward
    _Er: List[float]        # Sum of cumulative rewards
    _Er2: List[float]       # Sum of cycle cumulative reward squared
    _Elr: List[float]       # Sum of cycle product length and cycle
    _u: List[float]         # Estimated mean
    _Sm: List[float]        # Estimated variance
    _number_of_states: int  # length of distribution 

    def __init__(self, nr_of_states: int, confidence: float) -> None:
        self._number_of_states = nr_of_states
        self._l = 0 # Current cycle length
        self._Em = -1 # Cycle count (-1 to subtract unfinished cycle beforehand)
        self._El = 0 # Sum of cycle lengths
        self._El2 = 0 # Sum of cycle length squared
        self._rl = [0.0] * nr_of_states # Current cycle cumulative reward
        self._Er = [0.0] * nr_of_states # Sum of cumulative rewards
        self._Er2 = [0.0] * nr_of_states # Sum of cycle cumulative reward squared
        self._Elr = [0.0] * nr_of_states # Sum of cycle product length and cycle
        self._u = [0.0] * nr_of_states # Estimated mean
        self._Sm = [0.0] * nr_of_states # Estimated variance

        # Calculate the confidence point estimate with inverse normal distribution
        self._con = NormalDist().inv_cdf((1+confidence)/2)

    def visitRecurrentState(self)->None:
        self._Em += 1
        self._El += self._l
        self._El2 += pow(self._l, 2)
        self._Er = linalg.flAddVector(self._Er, self._rl)
        self._Er2 = linalg.flAddVector(self._Er2, [pow(r, 2) for r in self._rl])
        self._Elr = linalg.flAddVector(self._Elr, [r*self._l for r in self._rl])
        self._rl = [0.0] * self._number_of_states
        self._l = 0

        # Update point estimates of means.
        if self._El != 0:
            self._u = [er/self._El for er in self._Er]

        # Update point estimates of variances.
        if (self._El != 0) and (self._Em != 0):
            for i in range(len(self._Sm)):
                self._Sm[i] = math.sqrt(abs((self._Er2[i] - 2*self._u[i]*self._Elr[i] + pow(self._u[i],2)*self._El2)/self._Em))

    def addReward(self, n: int)->None:
        self._l += 1
        self._rl[n] += 1.0

    def cycleCount(self)->int:
        return self._Em

    def pointEstimates(self)->List[float]:
        return self._u

    def abErrorCezaro(self, noWarmup: bool = False)->List[Optional[float]]:
        '''Return estimated absolute errors'''
        # Run first _law times without checking abError
        abError: List[Optional[float]] = [None] * self._number_of_states

        if not noWarmup and (0 <= self._Em < _law):
            return abError

        if self._Em > 0:
            d = math.sqrt(float(self._Em)) * (1.0/float(self._Em)) * float(self._El)
            for i in range(len(abError)):
                if d != 0:
                    abError[i] = abs((self._con*self._Sm[i]) / d)
                else:
                    abError[i] = None
  
        return abError

    def reErrorCezaro(self, noWarmup: bool = False)->List[Optional[float]]:
        '''Return estimated relative errors'''
        reError: List[Optional[float]] = [None] * self._number_of_states

        # Run first .. times without checking abError
        if not noWarmup and (0 <= self._Em < _law):
            return reError
        
        for i in range(len(reError)):
            if self._Em > 0:
                d = (self._u[i] * math.sqrt(float(self._Em)) * (1.0/float(self._Em)) * self._El) - (self._con*self._Sm[i])
                if d != 0:
                    reError[i] = abs((self._con*self._Sm[i]) / d)
                else:
                    reError[i] = None

        return reError

    def confidenceIntervals(self)->List[Tuple[float,float]]:
        # Compute confidence interval
        abError = self.abErrorCezaro(noWarmup=True)
        return [(self._u[i] - abError[i], self._u[i] + abError[i]) for i in range(self._number_of_states)]

    def sanitizedReError(self)->List[Optional[float]]:
        reErrorVal = self.reErrorCezaro(noWarmup=True)
        abErrorVal = self.abErrorCezaro(noWarmup=True)
        result: List[Optional[float]] = []
        intervals = self.confidenceIntervals()
        for i in range(self._number_of_states):
            v: Optional[float] = reErrorVal[i]
            iv = intervals[i]
            # Check reError
            if (iv[0] < 0 and iv[1] >= 0) or reErrorVal[i] < 0:
                v = None
            
            # Check abError
            if abErrorVal[i] < 0: # Error value
                v = None
            result.append(v)
        return result

    def sanitizedAbError(self)->List[Optional[float]]:
        abErrorVal = self.abErrorCezaro(noWarmup=True)
        result: List[Optional[float]] = []
        for i in range(self._number_of_states):
            v: Optional[float] = abErrorVal[i]
            # Check abError
            if abErrorVal[i] < 0: # Error value
                v = None
            result.append(v)
        return result

    def sanitizedConfidenceIntervals(self)->List[Optional[Tuple[float,float]]]:
        abErrorVal = self.abErrorCezaro(noWarmup=True)
        result: List[Optional[Tuple[float,float]]] = []
        intervals = self.confidenceIntervals()
        for i in range(self._number_of_states):
            v: Optional[Tuple[float,float]] = intervals[i]          
            # Check abError
            if abErrorVal[i] < 0: # Error value
                v = None
            result.append(v)
        return result

    def sanitizedPointEstimates(self)->Optional[List[float]]:
        # Check if sum of distribution equals 1 with .4 float accuracy

        pntEst: List[float] = self.pointEstimates()
        if not 0.9999 < sum(pntEst) < 1.0001:
            return None
        return pntEst


# what is this for?

class WelfordWiki(object):
    '''
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    '''

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, newValue):
        # For a new value newValue, compute the new count, new mean, the new M2.
        # mean accumulates the mean of the entire dataset
        # M2 aggregates the squared distance from the mean
        # count aggregates the number of samples seen so far
        self._count += 1
        delta = newValue - self._mean
        self._mean += delta / self._count
        delta2 = newValue - self._mean
        self._M2 += delta * delta2

    def finalize(self, existingAggregate):
        # Retrieve the mean, variance and sample variance from an aggregate
        if self._count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sampleVariance)

class Welford(object):

    def __init__(self, confidence: float, stopDescriptions: List[str]) -> None:

        # confidence level
        self._c = NormalDist().inv_cdf((1+confidence)/2)

        self._interval: Tuple[float,float] = (0, 0)
        self._abErrorVal: float = -1.0
        self._reErrorVal: float = -1.0
        self._mean: float = 0.0
        self._count: int = 0
        self._M2: float = 0.0 # Welford's algorithm variable

        # There are in total four applicable stop conditions for this function
        self._y = 0.0

    def sample(self, v: float)->None:
        self._y = v
        # Execute Welford's algorithm to compute running standard derivation and mean
        self._count += 1
        d1 = self._y - self._mean
        self._mean += d1/self._count
        d2 = self._y - self._mean
        self._M2 += d1 * d2
        self._Sm = math.sqrt(self._M2/float(self._count))

    def getNumberOfSamples(self)->int:
        return self._count

    def getErrorVals(self)->Tuple[float,float]:

        # Compute absolute and relative errors
        self._abErrorVal = abs((self._c*self._Sm) / math.sqrt(float(self._count)))
        self._d = self._mean-self._abErrorVal
        if self._d != 0.0:
            self._reErrorVal = abs(self._abErrorVal/self._d)

        # interval calculation
        self._interval = (self._mean - self._abErrorVal, self._mean + self._abErrorVal)

        # Do not evaluate abError/reError in the first _law cycles:
        if self._count < _law and self._count != rounds: # (if rounds is less than 10 we still want an abError and reError)
            self._abErrorVal = -1.0
            self._reErrorVal = -1.0
        return self._abError, self._reError