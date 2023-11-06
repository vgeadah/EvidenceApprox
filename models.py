import numpy as np
import scipy as sp
from typing import Optional, Tuple

# Local imports
import utils

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
        self.A = 0.9 * utils.generate_rotation_matrix(latent_dim)
        self.B = 0.5 * np.random.rand(latent_dim, input_dim)
        self.C = 0.5 * np.random.rand(output_dim, latent_dim)
    
    def latent_forward(
            self, input: np.ndarray, latents_prev: Optional[np.ndarray]
            ) -> np.ndarray:
        '''
        sample z_t ~ p(z_t | z_{t-1}, u_t)
        '''
        if latents_prev is None:
            latents_prev = np.zeros(self.latent_dim)
        latents_next_mean = self.A @ latents_prev + self.B @ input
        latents_next = utils.reparameterize(
            loc = latents_next_mean,
            logscale = self.dynamics_lik_logscale
            )
        return latents_next
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        linear_decoder = self.C @ latent
        Y = utils.reparameterize(linear_decoder, logscale=self.likelihood_logscale)
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
    
    def marginal_log_likelihood(self, Y, U):
        '''
        p(y_{1:T})
        '''
        Q = np.exp(self.dynamics_lik_logscale) * np.eye(self.latent_dim)
        R = np.exp(self.likelihood_logscale) * np.eye(self.output_dim)

        # Run Kalman filter and save predicted state means and covariances
        (Xs_pred, Vs_pred), _ = utils.Kalman_filter(
            Y, U, self.A, self.B, self.C, Q=Q, R=R, 
            init_x=np.zeros(self.latent_dim), init_V=Q
            )

        # Compute log-likelihood in single forward pass
        log_lik = 0.
        for y, x_pred, V_pred in zip(Y, Xs_pred, Vs_pred):
            log_lik += sp.stats.multivariate_normal.logpdf(y, mean=self.C @ x_pred, cov=self.C @ V_pred @ self.C.T + R)

        return log_lik
    
    def sample(self, T, inputs=None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Sample from the model
        '''
        if inputs is None:
            inputs = np.zeros((T, self.input_dim))

        # Sample latent states
        latents = np.zeros((T, self.latent_dim))
        for t in range(T):
            if t==0:
                latents[t] = self.latent_forward(input=inputs[t], latents_prev=None)
            else:
                latents[t] = self.latent_forward(input=inputs[t], latents_prev=latents[t-1])
        
        # Sample observations
        Y = np.zeros((T, self.output_dim))
        for t in range(T):
            Y[t] = self.decode(latents[t])
        
        return Y, latents