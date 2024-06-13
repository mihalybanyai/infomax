import numpy as np
import matplotlib.pyplot as plt
import itertools

def observation_likelihood(observation, parameter):
    # Bernoulli
    return parameter if observation == 1 else (1-parameter) 

def sequence_likelihood(observations, parameter):
    return np.prod(np.array([observation_likelihood(o, parameter) for o in observations]))

def sequence_marginal(observations, parameter_prior):
    param_values = np.linspace(0, 1, len(parameter_prior))
    all_like = [sequence_likelihood(observations, param_values[i]) * parameter_prior[i] for i in range(len(parameter_prior))]
    return sum(all_like)

def mutual_information(parameter_prior, n_obs):
    param_values = np.linspace(0, 1, len(parameter_prior))
    obs_values = [list(i) for i in itertools.product([0, 1], repeat=n_obs)]
    mi = 0
    for o in range(len(obs_values)):
        act_marginal = sequence_marginal(obs_values[o], parameter_prior)
        act_log_marginal = np.log(act_marginal)
       #print(act_marginal, act_log_marginal)
        sub_mi = 0
        for th in range(len(param_values)):
            act_like = observation_likelihood(obs_values[o], param_values[th])
            if act_like > 0:
                sub_mi += act_like * (np.log(act_like) - act_log_marginal)
        mi += parameter_prior[th] * sub_mi
    return mi


parameter_resolution = 10
n_obs = 1

parameter_prior = np.ones(parameter_resolution) / parameter_resolution
print(mutual_information(parameter_prior, n_obs))

parameter_prior = np.zeros(parameter_resolution) 
parameter_prior[0] = 0.5
parameter_prior[-1] = 0.5
print(mutual_information(parameter_prior, n_obs))

#plt.bar(np.linspace(0, 1, parameter_resolution), parameter_prior, width=0.8 / parameter_resolution)
#plt.show()
