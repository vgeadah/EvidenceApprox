# Evidence approximation

Currently only supports marginal log-likelihood evaluationw with Sequential Monte Carlo (SMC, i.e. particle filtering) ([^1]: An Introduction to Sequential Monte Carlo. A. Doucet, N. de Freitas, N. Gordon.)

## Usage

The `bootstrap_filter` method in `samplers.py` is for general purpose. It can be used for any model of the form 
```
    z_t ~ p_t(z_t | z_{t-1}, input_t)         # Latents z_t
    y_t ~ p_t(y_t | z_t)                      # Emissions y_t
```

The user needs to specify two methods, with specifications:
```
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
```
and pass them to the `bootstrap_filter` method. See the `test_bootstrap()` function in `samplers.py` for an example use case in a Linear Gaussian Dynamical System model.