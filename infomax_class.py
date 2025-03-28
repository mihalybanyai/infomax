import numpy as np
import matplotlib.pyplot as plt
import itertools
import cProfile, pstats
import pickle
import os
from abc import ABC, abstractmethod

class distribution:
    def __init__(range):
        self.range = range
        self.eval_points = None
        self.prob_densities = None
        self.eval_num = None


class generative_model(ABC):

    def __init__(param_range):
        # TODO observation properties for later checks
        self.prior = distribution(range)

    @abstractmethod
    def observation_likelihood(observation, param_value):
        pass

    def sequence_likelihood(observations, param_value):
        # the probability of observing every observation in the sequence, given the parameter
        # product of the individual likelihoods, as observations are i.i.d.
        return np.prod(np.array([self.observation_likelihood(o, param_value) for o in observations]))

    def sequence_marginal(observations, prior):
        # probability of observing the sequence given the entire prior distribution of the parameter instead of one specific value
        all_like = [sequence_likelihood(observations, self.prior.eval_points[i]) * self.prior.prob_densities[i] for i in range(self.prior.eval_num)]
        return sum(all_like)

    def mutual_information(other_distr):
        # TODO finish this
        mi = 0
        for px in prob_densities:
            if px == 0: continue
            for py in other_distr.prob_densities:
                if py == 0: continue
                log_py = np.log(py)

                        
        for th in range(len(param_values)):
            # if the prior assigns 0 probability to the param value, it will not contribute to the MI
            if parameter_prior[th] == 0: continue
            # cycle over all possible observations        
            for o in range(len(obs_values)):
                if sequence_marginals[o] == 0: continue
                log_marginal = np.log(sequence_marginals[o])  # could be outside
                # if the probability of the observation given the parameter is 0, the contribution to the MI is 0 too
                act_like = sequence_likelihood(obs_values[o], param_values[th])
                if act_like > 0:
                    mi += parameter_prior[th] * act_like * (np.log(act_like) - log_marginal) 

