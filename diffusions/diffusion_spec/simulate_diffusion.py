

import jax
from jax import random
from jax import numpy as jnp
from jtils.batch_ops import batch_add, batch_mul
from ..typing import Function, KeyArray

def get_mean_scale_reverse_fn(score_fn: Function, 
                              mean_scale_fn: Function, 
                              eps: float = 1e-3, 
                              T: float=1.0, 
                              N: int=1000):
    
    def simulate_reverse_diffusion(rng: KeyArray, x_T: jnp.ndarray):
        shape = x_T.shape
        def update_fn(rng, score_fn, x_t, t):
            rng, step_rng= random.split(rng)
            x_mean, scale = mean_scale_fn(step_rng, x_t, t, score_fn)
            noise = random.normal(rng, x_t.shape)
            x = x_mean + batch_mul(scale, noise)
            return x, x_mean

        timesteps = jnp.linspace(T, eps, N)

        def loop_body(i, val):
            rng, x, x_mean = val
            t = timesteps[i]
            vec_t = jnp.ones(shape[0]) * t
            rng, step_rng = random.split(rng)
            x, x_mean = update_fn(step_rng, score_fn, x, vec_t)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, N, loop_body, (rng, x_T, x_T))

        return x, x_mean

    return simulate_reverse_diffusion
