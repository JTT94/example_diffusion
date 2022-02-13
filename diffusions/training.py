import jax
from jax import numpy as jnp
import flax
from typing import Any, Iterable
from .typing import KeyArray
from .model_utils import get_model_fn
from .model_ioputs import DiffusionModelInput
from .models.base import DiffusionModel


@flax.struct.dataclass
class State:
    step: int
    optimizer: flax.optim.Optimizer
    lr: float
    ema_rate: float
    ema_params: Any
    rng: Any


def clip_gradient(grad, grad_clip):
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
    clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad
    )
    return clipped_grad


def optimization_manager(warmup=0, grad_clip=-1):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(state, grad, warmup=warmup, grad_clip=grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        lr = state.lr
        if warmup > 0:
            lr = lr * jnp.minimum(state.step / warmup, 1.0)
        if grad_clip >= 0:
            clipped_grad = clip_gradient(grad, grad_clip)
        else:  # disabling gradient clipping if grad_clip < 0
            clipped_grad = grad
        return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

    return optimize_fn


def update_ema(ema_params, update_params, ema_rate):
    new_ema_params = jax.tree_multimap(
        lambda p_ema, p: p_ema * ema_rate + p * (1.0 - ema_rate),
        ema_params,
        update_params,
    )
    return new_ema_params


def get_loss_fn(model, sde):

    def loss_fn(rng, params, batch):
        model_fn = get_model_fn(model, params)
        
        # forward noising process
        x_0 = batch['x_0']
        perturbed_outputs = sde.forward_sample(rng, x_0)
        
        # compute score
        rng, step_rng = jax.random.split(rng)
        model_input = DiffusionModelInput(x_t=perturbed_outputs['x_t'], 
                                          x_0=perturbed_outputs['x_0'],
                                          t=perturbed_outputs['t'],
                                          rng=rng
                                         )
        
        score = model_fn(model_input)
        
        # compute loss pointwise through time
        loss = sde.pointwise_loss(score, perturbed_outputs)
        return loss

    return loss_fn


def get_step_fn(model, sde, optimize_fn):

    loss_fn = get_loss_fn(model, sde)

    def step_fn(carry_state, batch):
        rng, state = carry_state

        # gradient step
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)
        params = state.optimizer.target
        loss, grad = grad_fn(step_rng, params, batch)

        grad = jax.lax.pmean(grad, axis_name="batch")
        new_optimizer = optimize_fn(state, grad)

        # ema
        new_params_ema = update_ema(
            state.ema_params, new_optimizer.target, state.ema_rate
        )

        # update state
        step = state.step + 1
        new_state = state.replace(
            step=step, optimizer=new_optimizer, ema_params=new_params_ema
        )
        new_carry_state = (rng, new_state)
        loss = jax.lax.pmean(loss, axis_name="batch")
        return new_carry_state, loss

    return step_fn


def init_training_state(rng, model, input_shapes, optimizer, lr=1e-3, ema_rate=0.999):

    dummy_inputs = DiffusionModelInput(
        **{key: jnp.ones(shape) for key, shape in input_shapes.items()}
    )
    variables = model.init(rng, dummy_inputs)
    variables, initial_params = variables.pop("params")

    optimizer = optimizer.create(initial_params)

    state = State(
        step=0,
        rng=rng,
        ema_params=initial_params,
        ema_rate=ema_rate,
        lr=lr,
        optimizer=optimizer,
    )

    return state
