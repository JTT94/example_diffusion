import jax
from typing import Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, unfreeze

from transformers.file_utils import ModelOutput


@flax.struct.dataclass
class FlaxBaseModelOutput(ModelOutput):
    """
    
    """
    last_hidden_state: jnp.ndarray = jnp.array([0.1])
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None

if __name__ == "__main__":
    print(FlaxBaseModelOutput())