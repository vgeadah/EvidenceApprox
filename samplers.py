import numpy as np
import scipy as sp
from tqdm import tqdm

def bootstrap_filter(N: int, X, Y, model, seed=0):
    '''
    Bootstrap Filter / Particle filtering algorithm from [1], along with likelihood evaluation,
    specifically tailored for models regression models from X to Y.
    [1]: An Introduction to Sequential Monte Carlo. A. Doucet, N. de Freitas, N. Gordon.
    
    Args:
        N: int. Number of particles
        X: array, (T,). Inputs
        Y: array, (T,). Data/emissions. 
        model: needs to implement some form of forward method, and some form of emission likelihood
    Returns:
        z_history: List (T,), each element containing N samples, 
            Equally weighted latent samples to evalute integrals with MC methods
        loglik: scalar,
            Estimate of the marginal log-likelihood
    '''
    T = len(X)

    z_history = []
    z_t = np.zeros((N, model.latent_dim))
    loglik_running_estimate = 0.
    
    for t in tqdm(range(0,T), desc='Bootstrap filter'):

        # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
        #   Sample proposal N particles from previous N particles
        #   {tilde z_t^i, 1/N} is an approximation to p(z_t|y_{1:t-1})
        tilde_z_t = [model.latent_forward(input=X[t], latents_prev=_z_t) for _z_t in z_t]
        tilde_z_t = np.stack(tilde_z_t)

        # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
        #   {tilde z_t^i, tilde w^i} is an approximation to p(z_t|y_{1:t})
        tilde_w_t = model.emission_likelihood(y=Y[t], latent=tilde_z_t)
        
        # 3. Resampling step: 

        # 3.1. Calculate importance weights
        if np.sum(tilde_w_t) == 0.:
            choices = np.arange(N) # Nil likelihood, so no resampling
        else:
            # Normalize importance weights
            # normalized_tilde_w_t = sp.special.softmax(tilde_w_t) # Softmax
            normalized_tilde_w_t = tilde_w_t/np.sum(tilde_w_t) # L1 normalization

            choices = np.random.choice(N, size=N, p=normalized_tilde_w_t)
        
        # 3.2. Resample with replacement N particles according the importance weights
        # if t==0:
        z_t = np.array([tilde_z_t[c] for c in choices])
            # z_history = jnp.array([z_t])
        # else:
        #     # z_history.append(tilde_z_t)
        #     z_history = jnp.concatenate((z_history, z_t[jnp.newaxis,:]), axis=0)
        #     if isinstance(model, GLMLearn):
        #         temp = []
        #         for c in choices:
        #             # Select c'th trajectory
        #             # print([z_n.shape for z_n in z_history])
        #             _traj = np.array([z_n[c,:] for z_n in z_history]) # Length t
        #             # print(len(_traj))
        #             temp.append(_traj)
        #         z_history = jnp.transpose(jnp.stack(temp), (1,0,2))
        #         # print(_z_history.shape)
        #         # print([z_n.shape for z_n in z_history])
        #     else:
        #         raise NotImplementedError
        
        # lik_estimate = np.mean([model.emission_likelihood(y=Y[t], x_hat=_xhat, V=_V, sigma=sigma, softmax=softmax) for _xhat, _V in zip(xhat_t, V_t)])
        # lik_estimate = np.mean(model.emission_likelihood(y=Y[t], latent=z_t))
        lik_estimate = np.mean(tilde_w_t)
        if np.linalg.norm(lik_estimate) == 0.:
            loglik_running_estimate = np.nan
            break
        else:
            loglik_running_estimate += np.log(lik_estimate)

        z_history.append(z_t)

    return z_history, loglik_running_estimate

if __name__=='__main__':
    from models import LDS

    # Instantiate model
    input_dim = 2
    output_dim = 5
    latent_dim = 2
    model = LDS(latent_dim=latent_dim, input_dim=input_dim, output_dim=output_dim)

    # Generate data
    T = 10
    X = np.random.randn(T,input_dim)
    Y = np.random.randn(T,output_dim)

    # Run bootstrap filter
    N = 10
    z_history, loglik = bootstrap_filter(N=N, X=X, Y=Y, model=model)
    print(loglik)

    # Compare against true marginal log-likelihood
    loglik_true = model.marginal_log_likelihood(Y, U=X)
    print(loglik_true)

    # Compare with importance sampling from prior
    N_MC_samples = 1000
    log_lik_IS = 0.
    for _ in tqdm(range(N_MC_samples), 'Importance sampling'): 
        _, Z_samples = model.sample(T, inputs=X)
        log_lik_IS += np.sum(np.log([model.emission_likelihood(y=y, latent=z) for y, z in zip(Y, Z_samples)]))
    log_lik_IS /= N_MC_samples
    print(log_lik_IS)