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
        assert 1.0 - np.abs(np.sum(prob_densities)) < 1e-7
        self.eval_points = np.linspace(self.range[0], self.range[1], len(list(prob_densities)))
        self.prob_densities = prob_densities
        self.eval_num = len(prob_densities)
        self.bin_width = (self.range[1] - self.range[0]) / self.eval_num

    def sample(self, M):
        return np.random.choice(self.eval_points, size=M, p=self.prob_densities)

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
        # p(x | \theta)
        pass

    def observation_marginal(self, observation):
        # p(x) = \sum_\theta p(x | \theta) p(\theta)
        all_like = [self.observation_likelihood(observation, self.prior.eval_points[i]) * self.prior.prob_densities[i] for i in range(self.prior.eval_num)]
        return sum(all_like)

    def sequence_likelihood(self, observations, param_value):
        # p(X | theta) = \prod_x p(x | \theta)
        # the probability of observing every observation in the sequence, given the parameter
        # product of the individual likelihoods, as observations are i.i.d.
        return np.prod(np.array([self.observation_likelihood(o, param_value) for o in observations]))

    def sequence_marginal(self, observations):
        # p(X) = \sum_\theta p(X | \theta) p(\theta)
        # probability of observing the sequence given the entire prior distribution of the parameter instead of one specific value
        all_like = [self.sequence_likelihood(observations, self.prior.eval_points[i]) * self.prior.prob_densities[i] for i in range(self.prior.eval_num)]
        return sum(all_like)

    def _KL_components(self, N):
        # TODO implement the version using the prior samples
        storage_key = tuple(list(self.prior.prob_densities) + [N])
        if not storage_key in self.kl_components.keys():
            all_sequences = self.possible_observation_sequences(N)
            all_sequence_likelihoods = np.zeros((self.prior.eval_num, len(all_sequences)))  # p(X | \theta) \forall X, \theta
            for i_theta, p_theta in enumerate(self.prior.prob_densities):
                if p_theta == 0: continue
                for i_obs, obs in enumerate(all_sequences):
                    all_sequence_likelihoods[i_theta, i_obs] = self.sequence_likelihood(obs, self.prior.eval_points[i_theta])

            like_prior = all_sequence_likelihoods.transpose() * self.prior.prob_densities  # p(X | \theta) p(\theta) \forall X, \theta
            sequence_marginals = np.sum(like_prior.transpose(), axis=0)  # p(X) \forall X

            log_sequence_marginals = np.log(sequence_marginals, out=np.zeros_like(sequence_marginals, dtype=np.float64), where=(sequence_marginals!=0))
            log_sequence_likelihoods = np.log(all_sequence_likelihoods, out=np.zeros_like(all_sequence_likelihoods, dtype=np.float64), where=(all_sequence_likelihoods!=0))
            logdiff = log_sequence_likelihoods - log_sequence_marginals

            kl_components = all_sequence_likelihoods.transpose() * (logdiff.transpose())  # p(X | \theta) [\log p(X | \theta) - \log p(\theta)] \forall X, \theta
            self.kl_components[storage_key] = kl_components
        return self.kl_components[storage_key]
    
    def _observation_prob_ratios(self, observation):
        # p(x | \theta) / \sum_\theta p(x | \theta) p(\theta)
        likelihoods = np.array([self.observation_likelihood(observation, self.prior.eval_points[i]) for i in range(self.prior.eval_num)])
        marginal = np.sum(likelihoods * self.prior.prob_densities)
        return likelihoods / marginal
    
    def _predictive_distr(self, act_prior):
        possible_observations = list(range(self.n_possible_obs))
        predictive_probs = [sum([self.observation_likelihood(o, act_prior.eval_points[th]) * act_prior.prob_densities[th] for th in range(act_prior.eval_num)]) for o in possible_observations]
        return distribution(possible_observations, predictive_probs)

    def KL_divergences(self, N, posterior=None, M=0):
        future_KLs = np.nansum(self._KL_components(N), axis=0)  # KL[p(X | \theta) || p(X)] \forall \theta
        if posterior is None:
            return future_KLs
        else:
            # TODO why is this sometimes negative?
            # take M samples from the posterior-predictive distribution \sum_\theta p(x | \theta) p(\theta | X_old)
            samples = self._predictive_distr(posterior).sample(M)
            print(samples)
            sample_prob_ratios = np.array([self._observation_prob_ratios(s) for s in samples])  # M x theta_res
            print("ratio", sample_prob_ratios)
            log_sample_prob_ratios = np.log(sample_prob_ratios, out=np.zeros_like(sample_prob_ratios, dtype=np.float64), where=(sample_prob_ratios!=0))
            print("log", log_sample_prob_ratios)
            print("product", sample_prob_ratios * log_sample_prob_ratios)
            return np.sum(sample_prob_ratios * (log_sample_prob_ratios + future_KLs), axis=0)

    def mutual_information(self, N, posterior=None, M=0):
        # TODO implement the version with sampling
        return np.nansum(self.prior.prob_densities * self._KL_components(N))

    def blahut_arimoto_prior(self, N, prior_res, n_step, posterior=None, M=0, min_delta=0, plot=False):
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

    def posterior(self, observations):
        if self.prior.prob_densities is None:
            raise RuntimeError("Prior not set, cannot calculate posterior.")
        unnorm_post = np.array([self.sequence_likelihood(observations, self.prior.eval_points[th]) * self.prior.prob_densities[th] for th in range(self.prior.eval_num)])
        post = unnorm_post / np.sum(unnorm_post)
        return distribution(self.prior.range, post)

    def predictive_accuracy(self, true_probs, N):
        # get all sequences
        all_sequences = self.possible_observation_sequences(N)
        # calculate the true probability with which each sequence comes up
        true_seq_probs = np.array([np.prod(np.array([true_probs[obs] for obs in seq])) for seq in all_sequences])
        # get the posterior for each sequence, and set it as prior
        kl_score = 0
        for seq_idx, seq in enumerate(all_sequences):
            post = self.posterior(seq)
            # calculate the marginal likelihood of each single outcome 
            marginal_likelihood = np.zeros(self.n_possible_obs)
            for outcome in range(self.n_possible_obs):
                marginal_likelihood[outcome] = np.sum(np.array([self.observation_likelihood(outcome, post.eval_points[p]) * post.prob_densities[p] for p in range(post.eval_num)]))
            # calculate some scores, e.g. KL from true distr, weighted by true obs. prob.            
            act_kl =0
            for idx, px in enumerate(true_probs):
                py = marginal_likelihood[idx]
                if px > 0 and py > 0:
                    act_kl += px * (np.log(px) - np.log(py))  # TODO could be the other way around
            kl_score += act_kl * true_seq_probs[seq_idx]
            #print(marginal_likelihood, true_seq_probs[seq_idx], act_kl)
        return kl_score


class biased_coin_GM(generative_model):
    def __init__(self):
        super().__init__((0,1), 2)

    def observation_likelihood(self, observation, parameter):
        # the probability of observing this 1 value given the parameter value
        # Bernoulli distribution
        return parameter if observation == 1 else (1-parameter) 


"""
gm = biased_coin_GM()
gm.set_prior(prob_densities=np.ones(5)/5)
gm._observation_prob_ratios(1)

x = [0, 1, 1, 1, 0]
theta_res = 51
gm.blahut_arimoto_prior(len(x), theta_res, 1000, 1e-7, plot=False)
plt.subplot(2, 1, 1)
gm.posterior(x, plot=True)

gm2 = biased_coin_GM()
gm2.set_prior(np.ones(theta_res) / theta_res)
plt.subplot(2, 1, 2)
gm2.posterior(x, plot=True)

plt.show()


true_probs = [0.9999, 0.0001]
N = 4
theta_res = 51

gm = biased_coin_GM()
gm.blahut_arimoto_prior(N, theta_res, 1000, 1e-7, plot=False)
print(gm.predictive_accuracy(true_probs, N))

gm2 = biased_coin_GM()
gm2.set_prior(np.ones(theta_res) / theta_res)
print(gm2.predictive_accuracy(true_probs, N))"""
