import imp
import jax
from jax import numpy as jnp
from jax import random
from jtils.batch_ops import batch_mul
from ..model_ioputs import DiffusionOutput, DiffusionModelOutput, DiffusionModelInput
from .base import ReverseDiffusionSpec
from .losses import mean_squared_score_loss
from ..typing import KeyArray, Function
from .simulate_diffusion import get_mean_scale_reverse_fn


class DiscreteOU(ReverseDiffusionSpec):
    """ """

    def __init__(self, beta_min=0.1, beta_max=20, N=1000, eps=1e-3, T=1.0) -> None:
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.T = T
        self.eps = eps

        self.discrete_betas = jnp.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def sample_t(self, rng: KeyArray, x_0: jnp.ndarray) -> jnp.ndarray:
        return jax.random.choice(rng, self.N, (x_0.shape[0],))

    def mean_scale_function(
        self, rng: KeyArray, x_t: jnp.ndarray, t: jnp.ndarray, score_fn: Function
    ) -> jnp.ndarray:
        timestep = t * (self.N - 1) / self.T
        t_label = timestep.astype(jnp.int32)
        beta = self.discrete_betas[t_label]
        inputs = DiffusionModelInput(x_t=x_t, t=timestep)
        model_pred = score_fn(inputs)
        model_pred = model_pred.target
        std = self.sqrt_1m_alphas_cumprod[t_label.astype(jnp.int32)]
        score = batch_mul(-model_pred, 1.0 / std)
        x_mean = batch_mul((x_t + batch_mul(beta, score)), 1.0 / jnp.sqrt(1.0 - beta))
        return x_mean, jnp.sqrt(beta)

    def sample_x(self, rng: KeyArray, x_0: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        noise = random.normal(rng, x_0.shape)
        x_t = batch_mul(self.sqrt_alphas_cumprod[t], x_0) + batch_mul(
            self.sqrt_1m_alphas_cumprod[t], noise
        )
        return DiffusionOutput(x_t=x_t, z=noise, t=t)

    def pointwise_loss(
        self, model_output: DiffusionModelOutput, perturbed_output: DiffusionOutput
    ) -> jnp.ndarray:
        return mean_squared_score_loss(
            model_output=model_output, perturbed_output=perturbed_output
        )

    def get_simulate_reverse_diffusion_fn(self, score_fn: Function):
        return get_mean_scale_reverse_fn(
            score_fn, self.mean_scale_function, self.eps, self.T, self.N
        )

    def simulate_reverse_diffusion(
        self, rng: KeyArray, x_T: jnp.ndarray, score_fn: Function
    ):
        return get_mean_scale_reverse_fn(
            score_fn, self.mean_scale_function, self.eps, self.T, self.N
        )(rng, x_T)
