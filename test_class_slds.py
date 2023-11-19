"""
Simulate toy data from the SLDS model.
"""
import numpy as np
from models import SLDS
import matplotlib.pyplot as plt
import utils

def initialize_dynamics(K: int, M: int) -> dict:
    As = np.array([utils.generate_rotation_matrix(M) for _ in range(K)])
    bs = np.random.randn(K, M)
    Qs = np.array([np.eye(M) for _ in range(K)]) # todo: Make a random psd matrix


def initialize_emissions(N: int, M: int) -> dict:
    C = np.random.randn(N, M)
    d = np.random.randn(N)
    R = np.eye(N) # todo: Make a random psd matrix
    return {'C': C, 'd': d, 'R': R}


def initialize_transitions(K: int) -> dict:
    Ps = .95 * np.eye(K) + .05 * np.random.rand(K, K)
    Ps /= Ps.sum(axis=1, keepdims=True)
    return {'logPs': np.log(Ps)}


# Set the parameters of the SLDS
num_samples = 100    # number of time bins
K = 5       # number of discrete states
M = 2       # number of latent dimensions
N = 10      # number of observed dimensions



