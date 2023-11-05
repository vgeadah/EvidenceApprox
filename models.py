import numpy as np
import scipy as sp
from typing import Optional

def reparameterize(
        loc: np.ndarray, logscale: np.ndarray
    ) -> np.ndarray:
    '''Additive gaussian noise, for reparametrization trick.
    Returns:
        A sample from N(loc, logscale) of the same size as `loc`  
    '''
    scale = np.multiply(np.exp(logscale), np.ones_like(loc))
    eps = np.random.randn(*loc.shape)
    return loc + np.multiply(scale, eps)

class LDS():
    '''Linear Gaussian Dynamical System model'''
    def __init__(
            self, 
            latent_dim: int, input_dim: int, output_dim: int,
            dynamics_lik_logscale: float=-1.0, likelihood_logscale=-1.0,
            ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.dynamics_lik_logscale = dynamics_lik_logscale
        self.likelihood_logscale = likelihood_logscale
        
        # Define and initialize parameters
        self.A = 0.5*np.eye(latent_dim)
        self.B = np.random.rand(latent_dim, input_dim)
        self.C = np.random.rand(output_dim, latent_dim)
    
    def latent_forward(
            self, input: np.ndarray, latents_prev: Optional[np.ndarray]
            ) -> np.ndarray:
        '''
        sample z_t ~ p(z_t | z_{t-1}, u_t)
        '''
        if latents_prev is None:
            # Assume zero mean, + inputs
            latents_next_mean = self.B @ input
        else:
            latents_next_mean = self.A @ latents_prev + self.B @ input
        latents_next = reparameterize(
            loc = latents_next_mean,
            logscale = self.dynamics_lik_logscale
            )
        return latents_next
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        linear_decoder = self.C @ latent
        Y = reparameterize(linear_decoder, logscale=self.likelihood_logscale)
        return Y
    
    def emission_likelihood(self, y: np.ndarray, latent: np.ndarray) -> np.ndarray:
        '''
        p(y_t | z_t)
        '''
        Y = np.tile(y, (latent.shape[0], 1))
        linear_decoder = (self.C @ latent.T).T
        lik = np.array([
            sp.stats.multivariate_normal.pdf(y, mean=Cx, cov=np.exp(self.likelihood_logscale)) 
            for Cx in linear_decoder
            ])
        return lik