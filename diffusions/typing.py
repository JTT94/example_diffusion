from typing import Any, Union
from jax._src import prng
from flax.core.scope import FrozenVariableDict
from typing import Callable
from argparse import Namespace


Array = Any
Params = FrozenVariableDict
Function = Callable
KeyArray = Union[Array, prng.PRNGKeyArray]
