import numpy as np
import scipy as sp
from typing import Optional, Tuple

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

def generate_rotation_matrix(N):
    # Generate a random orthogonal matrix (Q) using the QR decomposition
    A = np.random.rand(N, N)
    Q, _ = np.linalg.qr(A)

    # Ensure that the determinant of Q is 1 (to make it a rotation)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q

def Kalman_filter(
        Y: np.ndarray, U: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray,
        init_x: np.ndarray, init_V: np.ndarray
        ):
    '''
    #TODO: Generated using copilot. Check. 

    Runs the Kalman smoother on a time series of observations y, with state transition matrix A,
    observation matrix C, process noise covariance Q, and observation noise covariance R.
    Assumes that the initial state x_0 is Gaussian with mean init_x and covariance init_V.
    Returns the smoothed state means and covariances.
    '''
    T, _ = Y.shape
    n_states = init_x.shape[0]

    # Initialize filtered and predictied state means and covariances
    Xs_pred = np.zeros((T, n_states))
    Vs_pred = np.zeros((T, n_states, n_states))
    Xs_filt = np.zeros((T, n_states))
    Vs_filt = np.zeros((T, n_states, n_states))

    # Initialize Kalman filter state means and covariances
    x_pred = init_x
    V_pred = init_V
    x_filt = init_x
    V_filt = init_V

    # Run Kalman filter forward pass
    for t in range(T):
        # Predict step
        x_pred = A @ x_filt + B @ U[t]
        V_pred = A @ V_filt @ A.T + Q

        # Update step
        y_t = Y[t]
        _cov = C @ V_pred @ C.T + R
        K_t = V_pred @ C.T @ np.linalg.inv(_cov)
        x_filt = x_pred + K_t @ (y_t - C @ x_pred)
        V_filt = (np.eye(n_states) - K_t @ C) @ V_pred
        # V_filt = V_pred - K_t @ (C @ V_pred @ C.T + R) @ K_t.T

        # Save filtered state means and covariances
        Xs_pred[t] = x_pred
        Vs_pred[t] = V_pred
        Xs_filt[t] = x_filt
        Vs_filt[t] = V_filt

    return (Xs_pred, Vs_pred), (Xs_filt, Vs_filt)

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
        self.A = 0.9 * generate_rotation_matrix(latent_dim)
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
    
    def marginal_log_likelihood(self, Y, U):
        '''
        p(y_{1:T})
        '''
        Q = np.exp(self.dynamics_lik_logscale) * np.eye(self.latent_dim)
        R = np.exp(self.likelihood_logscale) * np.eye(self.output_dim)

        # Run Kalman filter and save predicted state means and covariances
        (Xs_pred, Vs_pred), _ = Kalman_filter(
            Y, U, self.A, self.B, self.C, Q=Q, R=R, 
            init_x=np.zeros(self.latent_dim), init_V=Q
            )

        # Compute log-likelihood in single forward pass
        log_lik = 0.
        for y, x_pred, V_pred in zip(Y, Xs_pred, Vs_pred):
            # _cov = self.C @ V_pred @ self.C.T + R
            # _t1 = np.log(np.linalg.det(2*np.pi * _cov))
            # _t2 = (y - self.C @ x_pred).T @ np.linalg.inv(_cov) @ (y - self.C @ x_pred)
            # log_lik += -0.5 * (_t1 + _t2)

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