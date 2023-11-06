import numpy as np
import scipy as sp
from tqdm import tqdm
from typing import Callable, Optional, Tuple

def bootstrap_filter(
        N: int, 
        Y: np.ndarray, 
        X: Optional[np.ndarray], 
        latent_forward: Callable[[int, int, np.ndarray, np.ndarray], np.ndarray], 
        emission_likelihood: Callable[[int, int, np.ndarray, np.ndarray], np.ndarray], 
        return_history: bool=False
        ) -> Tuple[np.ndarray, float]:
    '''
    Bootstrap Filter / Particle filtering algorithm from [1], along with likelihood evaluation,
    for general models of the form 

        z_t ~ p_t(z_t | z_{t-1}, input_t)         # Latents z_t
        y_t ~ p_t(y_t | z_t)                      # Emissions y_t

    [1]: An Introduction to Sequential Monte Carlo. A. Doucet, N. de Freitas, N. Gordon.
    
    Args:
        N: int. number of particles
        X: array, (T, input_dim). inputs
        Y: array, (T, output_dim). data/emissions. 
        
        latent_forward: function, (t, input, latents_prev) -> latents_next
            Function to sample next N latents p(z_t | z_{t-1}, input_t) from N previous latents.
            Args:
                t: int, time step
                input: array, (input_dim,)
                latents_prev: array, (N, latent_dim)
            Returns:
                latents_next: array, (N, latent_dim)
       
        emission_likelihood: function, (N, t, y, latent) -> likelihood
            Function to evaluate the emission likelihood p(y_t | z_t) for N latents.
            Args:
                t: int, time step
                y: array, (output_dim,)
                latent: array, (N, latent_dim)
            Returns:
                likelihood: array, (N,)
    Returns:
        z_history: array (T, N, latent_dim),
            `N` equally weighted latent sample trajectories to evalute integrals with MC methods. 
            Returns empty list if return_history=False.
        loglik: scalar,
            Estimate of the marginal log-likelihood
    '''
    # Check inputs
    T = len(Y)
    if X is None:
        X = [None for _ in range(T)]

    # Initialize particles
    z_history = []
    z_t = [None for _ in range(N)]
    loglik_running_estimate = 0.
    
    # Run forward boostrap filter
    for t in tqdm(range(0,T), desc='Bootstrap filter'):

        # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1}, inputs_t)
        #    Sample proposal N particles from previous N particles
        #    {tilde z_t^i, 1/N} is an approximation to p(z_t|y_{1:t-1})
        tilde_z_t = latent_forward(t=t, input=X[t], latents_prev=z_t)

        # 2. Evaluate importance weights p(y_t | tilde z_t)
        tilde_w_t = emission_likelihood(t=t, y=Y[t], latent=tilde_z_t)
        
        # 3. Resampling step: 

        # 3.1. Choose trajectories from importance weights
        if np.sum(tilde_w_t) == 0.:
            choices = np.arange(N) # Nil likelihood, so no resampling
        else:
            normalized_tilde_w_t = tilde_w_t/np.sum(tilde_w_t) # L1 normalization
            # normalized_tilde_w_t = sp.special.softmax(tilde_w_t) # Softmax. Faster and differentiable, but not exact.

            choices = np.random.choice(N, size=N, p=normalized_tilde_w_t)
        
        # 3.X. Store log-likelihood estimate
        lik_estimate = np.mean(tilde_w_t)
        if np.linalg.norm(lik_estimate) == 0.:
            loglik_running_estimate = np.nan
            raise ValueError('Likelihood estimate is zero')
        else:
            loglik_running_estimate += np.log(lik_estimate)

        # 3.2. Resample with replacement N particles according the importance weights
        # z_history of shape (N, t+1, latent_dim)
        _, latent_dim = tilde_z_t.shape

        if return_history:
            if t==0:
                z_history = np.array([tilde_z_t[c] for c in choices])
                z_history = z_history.reshape(N, 1, latent_dim)
            else:
                # z_history of shape (N, t+1, latent_dim)
                z_history = np.concatenate((z_history, tilde_z_t[:, np.newaxis, :]), axis=1)
                z_history = np.array([z_history[c] for c in choices])
            z_t = z_history[:,-1,:]
        else:
            z_t = np.array([tilde_z_t[c] for c in choices])

    return z_history, loglik_running_estimate

def test_bootstrap():
    from models import LDS
    np.random.seed(1)

    # Instantiate model
    input_dim = 2
    output_dim = 5
    latent_dim = 2
    model = LDS(latent_dim=latent_dim, input_dim=input_dim, output_dim=output_dim)

    # Generate data
    T = 10
    X = np.random.randn(T,input_dim)
    Y = np.random.randn(T,output_dim)

    # Bootstrap filter
    N = 1000

    # Define appropriate functions for boostrap filter method
    def latent_forward(t, input, latents_prev):
        out = []
        for _z_t in latents_prev:
            if _z_t is None:
                _z_t = np.zeros(model.latent_dim)
            tilde_z_t = model.latent_forward(input=input, latents_prev=_z_t)
            out.append(tilde_z_t)
        out = np.stack(out)
        return out
    
    def emission_likelihood(t, y, latent):
        pyz = model.emission_likelihood(y=y, latent=latent)
        return pyz
    
    # Evaluate with bootstrap filter
    _, loglik = bootstrap_filter(
        N=N, Y=Y, X=X, 
        latent_forward=latent_forward, emission_likelihood=emission_likelihood,
        return_history=True
        )
    print(f'Bootstrap filter estimate: {loglik:.4f}')

    # Compare against true value
    loglik_true = model.marginal_log_likelihood(Y, U=X)
    print(f'True: {loglik_true:.4f}')

    # # Compare with MC evaluation with samples from prior
    # N_MC_samples = 1000
    # log_lik_IS = 0.
    # for _ in tqdm(range(N_MC_samples), 'Monte Carlo evaluation'): 
    #     _, Z_samples = model.sample(T, inputs=X)
    #     log_lik_IS += np.sum(np.log([model.emission_likelihood(y=y, latent=z) for y, z in zip(Y, Z_samples)]))
    # log_lik_IS /= N_MC_samples
    # print(log_lik_IS)

if __name__=='__main__':
    test_bootstrap()