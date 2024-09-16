import numpy as np
import matplotlib.pyplot as plt
import itertools
import cProfile, pstats
import pickle
import os

def observation_likelihood(observation, parameter):
    # the probability of observing this 1 value given the parameter value
    # Bernoulli distribution
    return parameter if observation == 1 else (1-parameter) 

def sequence_likelihood(observations, parameter):
    # the probability of observing every observation in the sequence, given the parameter
    # product of the individual likelihoods, as observations are i.i.d.
    return np.prod(np.array([observation_likelihood(o, parameter) for o in observations]))

def sequence_marginal(observations, parameter_prior):
    # probability of observing the sequence given the entire prior distribution of the parameter instead of one specific value
    param_values = np.linspace(0, 1, len(parameter_prior))
    all_like = [sequence_likelihood(observations, param_values[i]) * parameter_prior[i] for i in range(len(parameter_prior))]
    return sum(all_like)

def mutual_information(parameter_prior, n_obs, db=None):
    # TODO don't I calculate all likelihoods twice here?
    # TODO save / reload

    if not db is None and (tuple(parameter_prior), n_obs) in db.keys():
        mi = db[(tuple(parameter_prior), n_obs)]
    else:
        param_values = np.linspace(0, 1, len(parameter_prior))
        obs_values = [list(i) for i in itertools.product([0, 1], repeat=n_obs)]
        mi = 0
        
        sequence_marginals = [sequence_marginal(obs_values[o], parameter_prior) for o in range(len(obs_values))]
        # cycle over all possible parameter values
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
    if not db is None:
        db[(tuple(parameter_prior), n_obs)] = mi
        return mi, db
    else:
        return mi


parameter_resolution = 10
n_obs = 10

# 100 particles into 10 bins: 10^100 possibilities - not so nice

# what is a reasonable basis for this? one could just start from the uniform, and move one particle at a time, 
# that's roughly 90 possibilities in each step. to arrive to a fully binary distribution, one would need at least 80 steps, 
# so 7200 evaluations of the mutual information objective. this is a lower bound, but does not seem so horrifying. 
# profiling is definitely in order though. computation time will be heavily dependent on sequence length as well 

initialParticlesPerBin = 2
numberOfParticles = parameter_resolution * initialParticlesPerBin
particleQuantum = 1. / numberOfParticles
# print(particleQuantum)
parameter_prior = np.ones(parameter_resolution) * (particleQuantum * initialParticlesPerBin)
# print(parameter_prior)
actMI = mutual_information(parameter_prior, n_obs)

mi_db = {}
if os.path.isfile("mi_db.pickle"):
    with open("mi_db.pickle", "rb") as file:
        mi_db = pickle.load(file)

maxOptimStep = 20
printRes = 1
for i in range(maxOptimStep):
    if i % printRes == 0:
        print("optimisation step", i+1, "/", maxOptimStep)
    maxMI = actMI
    maxPrior = np.array(parameter_prior)
    for sourceBin in range(parameter_resolution):
        if parameter_prior[sourceBin] == 0.0: continue
        for targetBin in range(parameter_resolution):            
            if targetBin == sourceBin or parameter_prior[targetBin] == 1.1 : continue
            candidate_prior = np.array(parameter_prior)
            candidate_prior[sourceBin] -= particleQuantum
            candidate_prior[targetBin] += particleQuantum
            #print(candidate_prior)
            candidateMI, mi_db = mutual_information(candidate_prior, n_obs, mi_db)
            if candidateMI >= maxMI: 
                maxMI, maxPrior = candidateMI, np.array(candidate_prior)
    if maxMI == actMI:
        print("Local optimum reached after", i, "steps")
        break
    else:
        actMI, parameter_prior = maxMI, np.array(maxPrior)
        # print(actMI, parameter_prior)
print("Result of optimisation: MI=", actMI, parameter_prior)

with open("mi_db.pickle", "wb") as file:
    pickle.dump(mi_db, file)

# can I define an even simpler basis set? some series of deltas and then some near-deltas and at the end the uniform
# one problem is that the deltas are usually of different heights in the optimal prior. but maybe this can be disregarded 
# and the optimality sequence still retained
# probably a sequence of powers of 2, up to some number, and then the uniform


# profiler = cProfile.Profile()

# profiler.enable()
    
# parameter_prior = np.ones(parameter_resolution) / parameter_resolution
# print(mutual_information(parameter_prior, n_obs))

if n_obs == 1:
    parameter_prior = np.zeros(parameter_resolution) 
    parameter_prior[0] = 0.5
    parameter_prior[-1] = 0.5
    print("Optimal MI for 1 observation", mutual_information(parameter_prior, n_obs))

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats(30)
# profiler.dump_stats("agent.prof")

#(graph,) = pydot.graph_from_dot_file('callingGraph.dot')
#graph.write_png('callingGraph.png')

#plt.bar(np.linspace(0, 1, parameter_resolution), parameter_prior, width=0.8 / parameter_resolution)
#plt.show()
