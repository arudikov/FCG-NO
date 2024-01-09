import jax.numpy as jnp
import equinox as eqx

from jax import  config, random, vmap, tree_map
from jax.lax import dot_general
from jax.nn import relu, leaky_relu, hard_tanh, gelu
config.update("jax_enable_x64", True)

class FNO1D(eqx.Module):
    convs: list
    encoder_decoder: list
    A: list
    D: list
    N_modes: int

    def __init__(self, N_layers, N_features, N_modes, key):
        keys_A = random.split(key, N_layers+3)
        keys_conv = random.split(keys_A[-1], N_layers)
        keys_ed = random.split(keys_A[-2], 2)
        keys_D = random.split(keys_A[-3], N_layers)

        self.encoder_decoder = [eqx.nn.Conv(1, N_features[0], N_features[1], 1, key=keys_ed[0]), eqx.nn.Conv(1, N_features[1], N_features[2], 1, key=keys_ed[1])]
        self.convs = [eqx.nn.Conv(1, N_features[1], N_features[1], 1, key=key) for key in keys_conv]
        self.A = [random.normal(key, (N_features[1], N_features[1], N_modes), dtype=jnp.complex128) / N_features[1] for key in keys_A[:-3]]
        self.D = [random.normal(key, (N_features[1], N_features[1], 1)) / N_features[1] for key in keys_D]
        self.N_modes = N_modes

    def __call__(self, u):
        x = jnp.linspace(0,1,u.shape[-1])[None,...]
        u = jnp.concatenate([x, u], 0)
        u = self.encoder_decoder[0](u)
        for i in range(len(self.A)):
            v = self.convs[i](u)
            u = jnp.fft.rfft(u, axis=1)
            u = u.at[:, :1].set(self.ax(self.D[i] + 0j, u[:, :1]))
            u = u.at[:, 1:(self.N_modes+1)].set(self.ax(self.A[i], u[:, 1:(self.N_modes+1)]))
            u = u.at[:, (self.N_modes+1):].set(0)
            u = jnp.fft.irfft(u, axis=1, n=v.shape[1])
            u = u + v
            if i != (len(self.A) - 1):
                u = relu(u)
        u = self.encoder_decoder[1](u)
        return u

    def ax(self, A, x):
        return jnp.moveaxis(dot_general(A, x, (((1,), (0,)), ((2,), (1,)))), 1, 0)

class FNO2D(eqx.Module):
    convs: list
    encoder_decoder: list
    Ap: list
    An: list
    Dp: list
    Dn: list
    N_modes: int

    def __init__(self, N_layers, N_features, N_modes, key):
        keys_Ap = random.split(key, N_layers+5)
        keys_conv = random.split(keys_Ap[-1], N_layers)
        keys_ed = random.split(keys_Ap[-2], 2)
        keys_Dp = random.split(keys_Ap[-3], N_layers)
        keys_Dn = random.split(keys_Ap[-4], N_layers)
        keys_An = random.split(keys_Ap[-5], N_layers)

        self.encoder_decoder = [eqx.nn.Conv(2, N_features[0], N_features[1], 1, key=keys_ed[0]), eqx.nn.Conv(2, N_features[1], N_features[2], 1, key=keys_ed[1])]
        self.convs = [eqx.nn.Conv(2, N_features[1], N_features[1], 1, key=key) for key in keys_conv]
        self.Ap = [random.normal(key, (N_features[1], N_features[1], N_modes+1, N_modes), dtype=jnp.complex128) / N_features[1] for key in keys_Ap[:-5]]
        self.An = [random.normal(key, (N_features[1], N_features[1], N_modes, N_modes), dtype=jnp.complex128) / N_features[1] for key in keys_An]
        self.Dp = [random.normal(key, (N_features[1], N_features[1], N_modes+1, 1)) / N_features[1] for key in keys_Dp]
        self.Dn = [random.normal(key, (N_features[1], N_features[1], N_modes, 1)) / N_features[1] for key in keys_Dn]
        self.N_modes = N_modes

    def __call__(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder_decoder[0](u)
        for i in range(len(self.Ap)):
            v = self.convs[i](u)
            u = jnp.fft.rfftn(u, axes=[1, 2])
            u = u.at[:, :(self.N_modes+1), :1].set(self.ax(self.Dp[i] + 0j, u[:, :(self.N_modes+1), :1]))
            u = u.at[:, -self.N_modes:, :1].set(self.ax(self.Dn[i] + 0j, u[:, -self.N_modes:, :1]))

            u = u.at[:, :(self.N_modes+1), 1:(self.N_modes+1)].set(self.ax(self.Ap[i] + 0j, u[:, :(self.N_modes+1), 1:(self.N_modes+1)]))
            u = u.at[:, -self.N_modes:, 1:(self.N_modes+1)].set(self.ax(self.An[i] + 0j, u[:, -self.N_modes:, 1:(self.N_modes+1)]))
            u = u.at[:, :, (self.N_modes+1):].set(0)
            u = u.at[:, (self.N_modes+1):-self.N_modes, :].set(0)
            u = jnp.fft.irfftn(u, axes=[1, 2], s=v.shape[1:])
            u = u + v
            if i != (len(self.Ap) - 1):
                u = relu(u)
        u = self.encoder_decoder[1](u)
        return u

    def ax(self, A, x):
        return jnp.moveaxis(dot_general(A, x, (((1,), (0,)), ((2, 3), (1, 2)))), 2, 0)

def compute_loss(model, input, target):
    output = vmap(lambda z: model(z), in_axes=(0,))(input)
    l = jnp.mean(jnp.linalg.norm((output - target).reshape(input.shape[0], -1), axis=1)**2)
    return l / jnp.mean(jnp.linalg.norm((target).reshape(input.shape[0], -1), axis=1)**2)

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, input, target, optim, opt_state):
    # for optimizers that require model for update, e.g. adamw
    loss, grads = compute_loss_and_grads(model, input, target)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
