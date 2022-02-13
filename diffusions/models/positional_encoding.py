
import jax
from jax import numpy as jnp
import math
from flax import linen as nn

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def ff_embedding(seed, embedding_dim, input_dim, std):
    rng = jax.random.PRNGKey(seed)
    frequency_matrix = jax.random.normal(rng, (embedding_dim, input_dim))*jnp.sqrt(std)
    def embedder(coordinates):
        prefeatures = jnp.dot(coordinates, frequency_matrix.T)
        # Calculate cosine and sine features
        cos_features = jnp.cos(2 * jnp.pi * prefeatures)
        sin_features = jnp.sin(2 * jnp.pi * prefeatures)
        # Concatenate sine and cosine features
        return jnp.concatenate((cos_features, sin_features), axis=1)  
    return embedder

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  embed_dim: int
  scale: float = 30.
  @nn.compact
  def __call__(self, x):    
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), 
                 (self.embed_dim // 2, ))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

def fourier_encode(x: jnp.ndarray, num_encodings=4):
    x = jnp.expand_dims(x, -1)
    orig_x = x
    scales = 2 ** jnp.arange(num_encodings)
    x /= scales
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = jnp.concatenate([x, orig_x], axis=-1)
    return x