import numpy as np
import matplotlib.pyplot as plt
import itertools
import cProfile, pstats
import pickle
import os
from abc import ABC, abstractmethod

class distribution:
    def __init__(self, var_range, prob_densities=None):
        self.range = var_range
        self.eval_points = None
        self.prob_densities = None
        self.eval_num = None
        self.bin_width = None
        if not prob_densities is None:
            self.set_probs(prob_densities)

    def set_probs(self, prob_densities):
        # TODO why does this assert fail?
        #assert(np.abs(np.sum(prob_densities)) < 1e-5)
        self.eval_points = np.linspace(self.range[0], self.range[1], len(list(prob_densities)))
        self.prob_densities = prob_densities
        self.eval_num = len(prob_densities)
        self.bin_width = (self.range[1] - self.range[0]) / self.eval_num

    def plot(self):
        plt.bar(self.eval_points, self.prob_densities, width=self.bin_width * 0.8)


class generative_model(ABC):

    def __init__(self, param_range, n_possible_obs):
        # TODO generalise to continuous observations
        self.prior = distribution(param_range)
        self.n_possible_obs = n_possible_obs

        self.possible_sequences = {}  # TODO do this with a storage struct
        self.kl_components = {}
        # TODO load storage

    def set_prior(self, prob_densities):
        self.prior.set_probs(prob_densities)

    def possible_observation_sequences(self, N):
        if N in self.possible_sequences.keys():
            return self.possible_sequences[N]
        else:
            sequences = [list(i) for i in itertools.product(list(range(self.n_possible_obs)), repeat=N)]
            self.possible_sequences[N] = sequences
            return sequences

    @abstractmethod
    def observation_likelihood(self, observation, param_value):
        pass

    def sequence_likelihood(self, observations, param_value):
        # the probability of observing every observation in the sequence, given the parameter
        # product of the individual likelihoods, as observations are i.i.d.
        return np.prod(np.array([self.observation_likelihood(o, param_value) for o in observations]))

    def sequence_marginal(self, observations, prior):
        # probability of observing the sequence given the entire prior distribution of the parameter instead of one specific value
        all_like = [sequence_likelihood(observations, self.prior.eval_points[i]) * self.prior.prob_densities[i] for i in range(self.prior.eval_num)]
        return sum(all_like)

    def _KL_components(self, N):
        storage_key = tuple(list(self.prior.prob_densities) + [N])
        if not storage_key in self.kl_components.keys():
            all_sequences = self.possible_observation_sequences(N)
            all_sequence_likelihoods = np.zeros((self.prior.eval_num, len(all_sequences)))
            for i_theta, p_theta in enumerate(self.prior.prob_densities):
                if p_theta == 0: continue
                for i_obs, obs in enumerate(all_sequences):
                    all_sequence_likelihoods[i_theta, i_obs] = self.sequence_likelihood(obs, self.prior.eval_points[i_theta])

            like_prior = all_sequence_likelihoods.transpose() * self.prior.prob_densities
            sequence_marginals = np.sum(like_prior.transpose(), axis=0)
            #log_sequence_marginals = np.log(sequence_marginals)  # there will be nans
            log_sequence_marginals = np.log(sequence_marginals, out=np.zeros_like(sequence_marginals, dtype=np.float64), where=(sequence_marginals!=0))
            #log_sequence_likelihoods = np.log(all_sequence_likelihoods)  # there will be nans
            log_sequence_likelihoods = np.log(all_sequence_likelihoods, out=np.zeros_like(all_sequence_likelihoods, dtype=np.float64), where=(all_sequence_likelihoods!=0))
            logdiff = log_sequence_likelihoods - log_sequence_marginals
            kl_components = all_sequence_likelihoods.transpose() * (logdiff.transpose())
            self.kl_components[storage_key] = kl_components
        return self.kl_components[storage_key]

    def KL_divergences(self, N):
        return np.nansum(self._KL_components(N), axis=0)

    def mutual_information(self, N):
        return np.nansum(self.prior.prob_densities * self._KL_components(N))

    def blahut_arimoto_prior(self, N, prior_res, n_step, min_delta=0, plot=False):
        self.set_prior(np.ones(prior_res) / prior_res)
        MIs = [self.mutual_information(N)]

        for step in range(n_step):
            exp_kl = np.exp(self.KL_divergences(N))
            unnorm_new_p = exp_kl * self.prior.prob_densities
            self.set_prior(unnorm_new_p / np.sum(unnorm_new_p))
            MIs.append(self.mutual_information(N))
            if MIs[-1] - MIs[-2] <= min_delta:
                break

        if plot:
            plt.subplot(1, 2, 1)
            self.prior.plot()
            plt.subplot(1, 2, 2)
            plt.plot(MIs)

    def posterior(self, observations, plot=True):
        unnorm_post = np.array([self.sequence_likelihood(observations, self.prior.eval_points[th]) * self.prior.prob_densities[th] for th in range(self.prior.eval_num)])
        post = unnorm_post / np.sum(unnorm_post)
        post_distr = distribution((0, 1), post)
        if plot:
            post_distr.plot()



class biased_coin_GM(generative_model):
    def __init__(self):
        super().__init__((0,1), 2)

    def observation_likelihood(self, observation, parameter):
        # the probability of observing this 1 value given the parameter value
        # Bernoulli distribution
        return parameter if observation == 1 else (1-parameter) 


x = [0, 1, 1, 1, 0]
theta_res = 51

gm = biased_coin_GM()
gm.blahut_arimoto_prior(len(x), theta_res, 1000, 1e-7, plot=False)
plt.subplot(2, 1, 1)
gm.posterior(x, plot=True)

gm2 = biased_coin_GM()
gm2.set_prior(np.ones(theta_res) / theta_res)
plt.subplot(2, 1, 2)
gm2.posterior(x, plot=True)

plt.show()