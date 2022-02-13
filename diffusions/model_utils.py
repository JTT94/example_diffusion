from typing import Callable
import jax
from jax import numpy as jnp
from flax import linen as nn
from .typing import Params, Function
from .model_ioputs import DiffusionModelOutput
from .models.base import DiffusionModel

def get_model_fn(model: DiffusionModel, params: Params) -> Function:
    def model_fn(DiffusionModelInput) -> DiffusionModelOutput:
        model_output = model.apply({"params": params}, DiffusionModelInput)
        return model_output

    return model_fn
