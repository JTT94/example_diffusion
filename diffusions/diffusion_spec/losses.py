from jax import numpy as jnp
from ..model_ioputs import DiffusionOutput, DiffusionModelOutput


def mean_squared_score_loss(
    model_output: DiffusionModelOutput, perturbed_output: DiffusionOutput
) -> jnp.ndarray:

    noise = perturbed_output.z
    model_pred = model_output.target

    losses = jnp.square(model_pred - noise)
    losses = jnp.mean(losses.reshape((losses.shape[0], -1)), axis=-1)
    loss = jnp.mean(losses)

    return loss
