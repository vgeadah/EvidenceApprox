"""
Simulate toy data from the SLDS model.
"""
import numpy as np
from models import SLDS, Transitions, Dynamics, Emissions
import matplotlib.pyplot as plt
import utils

save_figures = True

def initialize_dynamics(K: int, M: int) -> dict:
    As = np.array([utils.generate_rotation_matrix(M) for _ in range(K)])
    bs = np.random.randn(K, M)
    Qs = np.array([np.eye(M) for _ in range(K)]) # todo: Make a random psd matrix
    return {'As': As, 'bs': bs, 'Qs': Qs}

def initialize_emissions(N: int, M: int) -> dict:
    C = np.random.randn(N, M)
    d = np.random.randn(N)
    R = np.eye(N) # todo: Make a random psd matrix
    return {'C': C, 'd': d, 'R': R}


def initialize_transitions(K: int) -> dict:
    Ps = .95 * np.eye(K) + .05 * np.random.rand(K, K)
    Ps /= Ps.sum(axis=1, keepdims=True)
    return {'logPs': np.log(Ps)}


if __name__ == "__main__":

    # Set the parameters of the SLDS
    num_samples = 100    # number of time bins
    K = 5       # number of discrete states
    M = 2       # number of latent dimensions
    N = 10      # number of observed dimensions

    trans_mat = initialize_transitions(K)
    dyn_mat = initialize_dynamics(K, M)
    emis_mat = initialize_emissions(N, M)

    print(trans_mat)
    print(dyn_mat)
    print(emis_mat)

    transitions = Transitions(K, trans_mat['logPs'])

    dynamics = Dynamics(M, K, dyn_mat['As'], 
                        dyn_mat['bs'], 
                        dyn_mat['Qs'])
    
    emissions = Emissions(N, M, emis_mat['C'],
                          emis_mat['d'], 
                          emis_mat['R'])


    slds = SLDS(M, N, K, transitions, dynamics, emissions)

    # Sample from the SLDS
    initial_state = {'z': 0, 'x': np.zeros((M, ))}
    latents, emissions = slds.sample(num_samples, initial_state)

    states_x = latents['x']
    states_z = latents['z']

    # Plot the latent states
    plt.figure(figsize=(10, 6))
    gs = plt.GridSpec(2, 1, height_ratios=(1, N/M))

    # Plot the continuous latent states
    lim = abs(states_x).max()
    plt.subplot(gs[0])
    for d in range(M):
        plt.plot(states_x[:, d] + lim * d, '-k')
    plt.yticks(np.arange(M) * lim, ["$x_{}$".format(d+1) for d in range(M)])
    plt.xticks([])
    plt.xlim(0, num_samples)
    plt.title("Simulated Latent States")

    lim = abs(emissions).max()
    plt.subplot(gs[1])
    for n in range(N):
        plt.plot(emissions[:, n] - lim * n, '-')
    plt.yticks(-np.arange(N) * lim, ["$y_{{ {} }}$".format(n+1) for n in range(N)])
    plt.xlabel("time")
    plt.xlim(0, num_samples)

    plt.title("Simulated Emissions")
    plt.tight_layout()

    if save_figures:
        plt.savefig("sim_SLDS.pdf")