import math
from statistics import NormalDist
from typing import List, Tuple
import markovchains.utils.linalgebra as linalg

# As a rule of thumb, a reasonable number of results are needed before certain calculations can be considered valid
# this variable determines the number of results that are required for the markov simulation before the stop conditions
# are checked. (Based on the law of strong numbers)
_law: int = 30


# TODO: modify terminology: reward to sample ?
class Statistics(object):

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

    def addReward(self, r: float)->None:
        self._l += 1
        self._r += r

    def cycleCount(self)->int:
        return self._Em

    def pointEstU(self):
        '''Update point estimate of mean.'''
        if self._El != 0:
            self._u = self._Er/self._El

    def pointEstSm(self):
        '''Update point estimate of variance.'''
        if (self._El != 0) and (self._Em != 0):
            self._Sm = math.sqrt(abs((self._Er2 - 2*self._u*self._Elr + pow(self._u,2)*self._El2)/self._Em))

    # TODO: why parameter n?
    def abError(self, n: int)->float:
        '''Return estimated absolute error'''
        # Run first _law times without checking abError
        if 0 <= n < _law:
            return -1.0
        if self._Em > 0:
            d = math.sqrt(self._Em) * (1/(self._Em)) * self._El
            if d != 0:
                return abs((self._con*self._Sm) / d)
            else:
                return -1.0
        else:
            return -1.0

    # TODO: why parameter n?
    def reError(self, n: int)->float:
        '''Return estimated relative error'''
        # Run first 10 times without checking abError
        if 0 <= n < _law:
            return -1.0

        if self._Em > 0:
            d = (self._u * math.sqrt(self._Em) * (1/(self._Em)) * self._El) - (self._con*self._Sm)
            if d != 0:
                return abs((self._con*self._Sm) / d)
            else:
                return -1.0
        else:
            return -1.0

    def confidenceInterval(self)->Tuple[float,float]:
        # Compute confidence interval
        abError = self.abError(_law)
        return (self._u - abError, self._u + abError)


class DistributionStatistics(object):

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

    def addReward(self, n: int)->None:
        self._l += 1
        self._rl[n] += 1.0

    def cycleCount(self)->int:
        return self._Em

    def pointEstUCezaro(self):
        '''Update point estimates of means.'''
        if self._El != 0:
            self._u = [er/self._El for er in self._Er]

    def pointEstimates(self)->List[float]:
        return self._u

    def pointEstSmCezaro(self):
        '''Update point estimates of variances.'''
        if (self._El != 0) and (self._Em != 0):
            for i in range(len(self._Sm)):
                self._Sm[i] = math.sqrt(abs((self._Er2[i] - 2*self._u[i]*self._Elr[i] + pow(self._u[i],2)*self._El2)/self._Em))

    def abErrorCezaro(self, n: int)->List[float]:
        '''Return estimated absolute errors'''
        # Run first _law times without checking abError
        abError = [-1.0] * self._number_of_states

        if 0 <= n < _law:
            return abError

        if self._Em > 0:
            d = math.sqrt(float(self._Em)) * (1.0/float(self._Em)) * float(self._El)
            for i in range(len(abError)):
                if d != 0:
                    abError[i] = abs((self._con*self._Sm[i]) / d)
                else:
                    abError[i] = -1.0
  
        return abError

    def reErrorCezaro(self, n: int)->List[float]:
        '''Return estimated relative errors'''
        reError = [-1.0] * self._number_of_states

        # Run first 10 times without checking abError
        if 0 <= n < _law:
            return reError
        
        for i in range(len(reError)):
            if self._Em > 0:
                d = (self._u[i] * math.sqrt(float(self._Em)) * (1.0/float(self._Em)) * self._El) - (self._con*self._Sm[i])
                if d != 0:
                    reError[i] = abs((self._con*self._Sm[i]) / d)
                else:
                    reError[i] = -1.0

        return reError

    def confidenceIntervals(self)->List[Tuple[float,float]]:
        # Compute confidence interval
        abError = self.abErrorCezaro(_law)
        return [(self._u[i] - abError[i], self._u[i] + abError[i]) for i in range(self._number_of_states)]
