import numpy as np

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