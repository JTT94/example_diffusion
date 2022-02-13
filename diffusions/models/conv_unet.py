import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
from typing import Any, Tuple
import functools
import jax
from ..typing import KeyArray
from ..model_ioputs import DiffusionModelInput, DiffusionModelOutput, ModelConfig
from .positional_encoding import GaussianFourierProjection, get_timestep_embedding
from .base import DiffusionModel

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    output_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)[:, None, None, :]


class ConvUnet(DiffusionModel):
    """A time-dependent score-based model built upon U-Net architecture.

    Args:
        marginal_prob_std: A function that takes time t and gives the standard
          deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
    """

    config: ModelConfig

    @nn.compact
    def __call__(self, model_input: DiffusionModelInput):
        embed_dim = self.config.embed_dim
        channels = self.config.channels

        x = model_input.x_t
        t = model_input.t

        # The swish activation function
        act = nn.relu
        # Obtain the Gaussian random feature embedding for t
        embed = act(
            nn.Dense(embed_dim)(get_timestep_embedding(t, embedding_dim=embed_dim))
        )

        # Encoding path
        h1 = nn.Conv(channels[0], (3, 3), (1, 1), padding="VALID", use_bias=False)(x)
        ## Incorporate information from t
        h1 += Dense(channels[0])(embed)
        ## Group normalization
        h1 = nn.GroupNorm(4)(h1)
        h1 = act(h1)
        h2 = nn.Conv(channels[1], (3, 3), (2, 2), padding="VALID", use_bias=False)(h1)
        h2 += Dense(channels[1])(embed)
        h2 = nn.GroupNorm()(h2)
        h2 = act(h2)
        h3 = nn.Conv(channels[2], (3, 3), (2, 2), padding="VALID", use_bias=False)(h2)
        h3 += Dense(channels[2])(embed)
        h3 = nn.GroupNorm()(h3)
        h3 = act(h3)
        h4 = nn.Conv(channels[3], (3, 3), (2, 2), padding="VALID", use_bias=False)(h3)
        h4 += Dense(channels[3])(embed)
        h4 = nn.GroupNorm()(h4)
        h4 = act(h4)

        # Decoding path
        h = nn.Conv(
            channels[2],
            (3, 3),
            (1, 1),
            padding=((2, 2), (2, 2)),
            input_dilation=(2, 2),
            use_bias=False,
        )(h4)
        ## Skip connection from the encoding path
        h += Dense(channels[2])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(
            channels[1],
            (3, 3),
            (1, 1),
            padding=((2, 3), (2, 3)),
            input_dilation=(2, 2),
            use_bias=False,
        )(jnp.concatenate([h, h3], axis=-1))
        h += Dense(channels[1])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(
            channels[0],
            (3, 3),
            (1, 1),
            padding=((2, 3), (2, 3)),
            input_dilation=(2, 2),
            use_bias=False,
        )(jnp.concatenate([h, h2], axis=-1))
        h += Dense(channels[0])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
            jnp.concatenate([h, h1], axis=-1)
        )
        h = h + x

        return DiffusionModelOutput(target=h)
