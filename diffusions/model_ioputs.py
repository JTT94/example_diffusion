import jax
from typing import Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, unfreeze
from collections import OrderedDict
from .typing import KeyArray
from jtils.namespace import dict_to_namespace
from argparse import Namespace
from jtils.namespace import namespace_to_dict


class DataClass(OrderedDict):
    def __getitem__(self, key):
        return self.__getattribute__(key)


@flax.struct.dataclass
class DiffusionOutput(DataClass):
    """ """

    x_t: jnp.ndarray = None
    x_0: Optional[jnp.ndarray] = None
    z: Optional[jnp.ndarray] = None
    t: Optional[jnp.ndarray] = None
    latent: Optional[jnp.ndarray] = None
    label: Optional[jnp.ndarray] = None


@flax.struct.dataclass
class DiffusionModelOutput(DataClass):
    """ """

    target: jnp.ndarray = None
    x_next: Optional[jnp.ndarray] = None
    x_0: Optional[jnp.ndarray] = None


@flax.struct.dataclass
class DiffusionModelInput(DataClass):
    """ """

    x_t: jnp.ndarray = None
    t: jnp.ndarray = None
    rng : Optional[KeyArray] = None
    x_0: Optional[jnp.ndarray] = None



class ModelConfig(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def items(self):
        return namespace_to_dict(self).items()

    def __getitem__(self, key):
        return self.__getattribute__(key)
