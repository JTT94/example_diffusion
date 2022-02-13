
import jax
from jax import numpy as jnp
from jax import random
from ..model_ioputs import DiffusionOutput, DiffusionModelOutput
from ..typing import KeyArray, Function

class ReverseDiffusionSpec:

    def sample_t(self, rng: KeyArray, x_0: jnp.ndarray) -> jnp.ndarray:
        pass

    def sample_x(self, x_0 : jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        pass

    def forward_sample(self, rng: KeyArray, x_0: jnp.ndarray) -> DiffusionOutput:
        rng, t_rng, x_rng = random.split(rng, 3)
        t = self.sample_t(t_rng, x_0)
        return self.sample_x(x_rng, x_0, t)

    def pointwise_loss(self, model_output: DiffusionModelOutput, perturbed_output: DiffusionOutput) -> jnp.ndarray:
        pass

    def simulate_reverse_diffusion(self, rng: KeyArray, x_T: jnp.ndarray, score_fn: Function):
        pass
