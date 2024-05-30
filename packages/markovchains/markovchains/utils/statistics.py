"""Statistical methods for simulation based estimation."""
import math
from statistics import NormalDist
from typing import List, Optional, Tuple, Union

# As a rule of thumb, a reasonable number of results are needed before
# certain calculations can be considered valid this variable determines
# the number of results that are required for the markov simulation
# before the stop conditions are checked. (Based on the law of strong
# numbers)
Minimum_Nr_Samples: int = 30

RES_TOO_FEW_SAMPLES = "Cannot be decided, too few samples."
class Statistics:
    '''
    Determine estimated long-run sample average, absolute error, relative
    error and confidence interval
    Equations follow Section B.7.3. of the Reader 5xie0, Computational Modeling
    '''

    _cycle_length:int                            # Current cycle length Ln
    _sum_of_samples_cycle: float                   # Current cycle, sum of samples
    _cycle_count: int                            # Cycle count
    _cumulative_cycle_lengths: int                # Sum of cycle lengths
    _cumulative_samples: float                   # Cumulative sum of samples
    _cumulative_cycle_lengths_sq: int              # Sum of cycle length squared
    _cumulative_samples_cycle_sq: float            # Sum of cycle cumulative samples squared
    _cumulative_prod_cycle_length_sum_samples: float # Sum of cycle product length and cycle
    _mean_est: float                             # Estimated mean
    _std_dev_est: float                           # Estimated variance
    _conf_level: float                           # confidence level
    _nr_paths: int                               # keep track of number of paths explored
    _minimum_nr_samples: int                      # minimum number of samples to use

    def __init__(self, confidence: float, minimum_number_of_samples: int = Minimum_Nr_Samples)\
          -> None:
        '''confidence: confidence level'''
        self._cycle_length = 0                             # Current cycle length
        self._sum_of_samples_cycle = 0.0                   # Current cycle cumulative samples
        # Cycle count (-1 to subtract unfinished cycle beforehand)
        self._cycle_count = 0
        self._cumulative_cycle_lengths = 0                 # Sum of cycle lengths
        self._cumulative_samples = 0.0                     # Sum of cumulative samples
        self._cumulative_cycle_lengths_sq = 0              # Sum of cycle length squared
        self._cumulative_samples_cycle_sq = 0              # Sum of cycle cumulative samples squared
        self._cumulative_prod_cycle_length_sum_samples = 0 # Sum of cycle product length and cycle
        self._mean_est = 0                                 # Estimated mean
        self._std_dev_est = 0                              # Estimated variance
        self._nr_paths = 0
        self._minimum_nr_samples = minimum_number_of_samples

        # Calculate the confidence point estimate with inverse normal distribution
        self._conf_level = NormalDist().inv_cdf((1+confidence)/2)

    def update_mean_estimate(self)->None:
        """Update point estimate of mean, Eq. B.66."""
        if self._cumulative_cycle_lengths != 0:
            self._mean_est = self._cumulative_samples/self._cumulative_cycle_lengths

    def mean_estimate(self)->Optional[float]:
        """Determine estimate of mean."""
        if self._cycle_count < self._minimum_nr_samples:
            return None
        return self._mean_est

    def mean_estimate_result(self)->Union[str,float]:
        """Get final mean estimate."""
        if self._cycle_count < self._minimum_nr_samples:
            return RES_TOO_FEW_SAMPLES
        return self._mean_est

    def std_dev_estimate(self)->Optional[float]:
        """Get standard deviation estimate."""
        if self._cycle_count < self._minimum_nr_samples:
            return None
        return self._std_dev_est

    def update_variance_estimate(self)->None:
        """Update the variance estimate."""
        # Compute S_M following the equation below Eq. B.67 of the reader
        # Update point estimate of standard deviation.
        # assumes mean estimate is up to date!
        if (self._cumulative_cycle_lengths != 0) and (self._cycle_count != 0):
            self._std_dev_est = math.sqrt(abs(
                self._cumulative_samples_cycle_sq
                - 2*self._mean_est*self._cumulative_prod_cycle_length_sum_samples
                + pow(self._mean_est,2)*self._cumulative_cycle_lengths_sq
            )/self._cycle_count)

    def complete_cycle(self)->None:
        """Complete a recurrence cycle in the estimation."""
        self._cycle_count += 1
        self._cumulative_cycle_lengths += self._cycle_length
        self._cumulative_samples += self._sum_of_samples_cycle
        self._cumulative_cycle_lengths_sq += pow(self._cycle_length, 2)
        self._cumulative_samples_cycle_sq += pow(self._sum_of_samples_cycle, 2)
        self._cumulative_prod_cycle_length_sum_samples += self._cycle_length * \
            self._sum_of_samples_cycle

        # reset accumulators
        self._cycle_length = 0
        self._sum_of_samples_cycle = 0.0

        self.update_mean_estimate()
        self.update_variance_estimate()

    def add_sample(self, s: float)->None:
        """Add a sample for estimation."""
        self._cycle_length += 1
        self._sum_of_samples_cycle += s

    def cycle_count(self)->int:
        """Get the current cycle count."""
        return self._cycle_count

    def ab_error(self)->Optional[float]:
        '''
        Return estimated absolute error.
        If no estimate can be given, None is returned
        '''

        # check if we have collected sufficient cycles
        if self._cycle_count < self._minimum_nr_samples:
            return None

        if self._cycle_count == 0:
            return None

        # denominator of the absolute error term in Eq. B.69
        den = self._cumulative_cycle_lengths/math.sqrt(self._cycle_count)
        if den != 0.0:
            return abs((self._conf_level*self._std_dev_est) / den)
        return None

    def ab_error_reached(self, max_ab_error: float)-> bool:
        """Check if absolute error bound is reached."""
        ab_error_val = self.ab_error()
        if ab_error_val is not None:
            return  0.0 <= ab_error_val <= max_ab_error
        return False

    def re_error(self)->Optional[float]:
        '''Return estimated relative error'''

        # check if we have collected sufficient cycles
        if 0 <= self._cycle_count < self._minimum_nr_samples:
            return None

        if self._cycle_count == 0:
            return None

        abs_error = self.ab_error()
        if abs_error is None:
            return None

        if abs_error >= abs(self._mean_est):
            return None

        return abs_error / (abs(self._mean_est) - abs_error)

    def re_error_reached(self, max_re_error: float)-> bool:
        """Check if relative error is reached."""
        re_error_val = self.re_error()
        if re_error_val is not None:
            return  0.0 <= re_error_val <= max_re_error
        return False

    def confidence_interval(self)->Optional[Tuple[float,float]]:
        """Compute confidence interval"""
        ab_error = self.ab_error()
        if ab_error is None:
            return None
        return (self._mean_est - ab_error, self._mean_est + ab_error)

    def increment_paths(self)->None:
        """Increment the number of paths."""
        self._nr_paths += 1

    def nr_paths(self)->int:
        """Return the number of paths."""
        return self._nr_paths


class DistributionStatistics:
    '''Determine Cesaro limit distribution'''

    _number_of_states: int  # length of distribution
    _state_estimators: List[Statistics]

    def __init__(self, nr_of_states: int, confidence: float) -> None:
        self._number_of_states = nr_of_states
        self._state_estimators = [Statistics(confidence) for i in range(nr_of_states)]

    def complete_cycle(self)->None:
        """Complete a recurrence cycle."""
        for s in self._state_estimators:
            s.complete_cycle()

    def visit_state(self, n: int)->None:
        """Process visiting of state."""
        for i in range(self._number_of_states):
            if i == n:
                self._state_estimators[i].add_sample(1.0)
            else:
                self._state_estimators[i].add_sample(0.0)

    def cycle_count(self)->int:
        """Return cycle count."""
        return self._state_estimators[0].cycle_count()

    def point_estimates(self)->Optional[List[float]]:
        """Return list of point estimates."""
        res = [s.mean_estimate() for s in self._state_estimators]
        if None in res:
            return None
        v_res: List[float] = res  # type: ignore
        if not 0.9999 < sum(v_res) < 1.0001:
            return None

        return v_res

    def std_dev_estimates(self)->Optional[List[float]]:
        """Get list of standard deviation estimates."""
        res = [s.std_dev_estimate() for s in self._state_estimators]
        if None in res:
            return None
        v_res: List[float] = res  # type: ignore
        return v_res

    def ab_error(self)->List[Optional[float]]:
        '''Return estimated absolute errors'''
        return [s.ab_error() for s in self._state_estimators]

    def max_ab_error(self)->Optional[float]:
        '''Return maximum estimated absolute error'''
        res = [s.ab_error() for s in self._state_estimators]
        if any(e is None for e in res):
            return None
        v_res: List[float] = res  # type: ignore
        return max(v_res)

    def ab_error_reached(self, max_ab_error: float)->bool:
        """Check if absolute error is reached."""
        ab_error = self.ab_error()
        if not None in ab_error:
            v_ab_error: List[float] = ab_error  # type: ignore
            return 0 <= max(v_ab_error) <= max_ab_error
        return False

    def re_error(self)->List[Optional[float]]:
        '''Return estimated relative errors'''
        return [s.re_error() for s in self._state_estimators]

    def max_re_error(self)->Optional[float]:
        '''Return maximum estimated relative error'''
        res = [s.re_error() for s in self._state_estimators]
        if any(e is None for e in res):
            return None
        v_res: List[float] = res  # type: ignore
        return max(v_res)

    def re_error_reached(self, max_re_error: float)->bool:
        """Check if relative error is reached."""
        re_error = self.re_error()
        if not None in re_error:
            v_re_error: List[float] = re_error  # type: ignore
            return 0 <= max(v_re_error) <= max_re_error
        return False

    def confidence_intervals(self)->Optional[List[Tuple[float,float]]]:
        """Return list of confidence intervals."""
        res = [s.confidence_interval() for s in self._state_estimators]
        if any(i is None for i in res):
            return None
        v_res: List[Tuple[float,float]] = res  # type: ignore
        return v_res

class StopConditions:
    """Stopping conditions"""

    confidence: float
    max_ab_error: float
    max_re_error: float
    max_path_length: int
    nr_of_cycles: int
    seconds_timeout: float
    minimum_number_of_samples: int

    def __init__(self, confidence: float,max_ab_error: float,max_re_error: float,max_path_length: \
                 int,nr_of_cycles: int,seconds_timeout: float, \
                    minimum_number_of_samples: int = Minimum_Nr_Samples) -> None:
        self.confidence = confidence
        self.max_ab_error = max_ab_error
        self.max_re_error = max_re_error
        self.max_path_length = max_path_length
        self.nr_of_cycles = nr_of_cycles
        self.seconds_timeout = seconds_timeout
        self.minimum_number_of_samples = minimum_number_of_samples
