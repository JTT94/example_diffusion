import jax.numpy as jnp
from flax import linen as nn
from ..model_ioputs import DiffusionModelInput, DiffusionModelOutput, ModelConfig


class DiffusionModel(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, model_input: DiffusionModelInput) -> DiffusionModelOutput:
        """[summary]

        Args:
            model_input (DiffusionModelInput): [description]

        Returns:
            DiffusionModelOutput: [description]
        """
        pass
