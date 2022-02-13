import jax.numpy as jnp
from flax import linen as nn
from ..model_ioputs import DiffusionModelInput, DiffusionModelOutput, ModelConfig
from .base import DiffusionModel
from .positional_encoding import get_timestep_embedding


class FCBlock(nn.Module):
    hidden_layer: int = 128
    num_layers: int = 3
    activation: nn.Module = nn.relu
    out_dim: int = 32

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_layer, name="fc{0}".format(i))(x)
            x = self.activation(x)

        x = nn.Dense(self.out_dim, name="fc_final")(x)
        return x


class MLPNet(DiffusionModel):

    config: ModelConfig

    @nn.compact
    def __call__(self, model_input: DiffusionModelInput):
        x_t = model_input.x_t
        t = model_input.t
        config = self.config
        out_dim = x_t.shape[1]
        t_emb = get_timestep_embedding(t, embedding_dim=config.t_pos_dim)
        t_emb = FCBlock(hidden_layer=config.t_embed_dim, out_dim=config.t_embed_dim)(
            t_emb
        )
        x_emb = FCBlock(hidden_layer=config.x_embed_dim, out_dim=config.x_embed_dim)(
            x_t
        )

        emb = jnp.concatenate([x_emb, t_emb], axis=-1)

        vec = FCBlock(hidden_layer=config.joint_hidden_dim, out_dim=out_dim)(emb)
        vec = vec + x_t

        return DiffusionModelOutput(target=vec)
