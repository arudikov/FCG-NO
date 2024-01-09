import jax.numpy as jnp
import equinox as eqx

from jax import config, random, vmap
from jax.lax import dot_general, slice
from jax.nn import relu
from typing import Callable
from architectures.DilResNet import DilatedConvBlock
from transforms import cheb

config.update("jax_enable_x64", True)

class ChebNO(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    conv: list
    axes: list
    processor: list
    weights: list
    N_pad: list
    N_cut: list

    def __init__(self, key, N_layers, N_features, kernel_size, N_conv, D, N_modes, weights):
        '''
        Example of initialization:

        key = random.PRNGKey(11)
        inp = random.normal(key, (8, 12, 14, 27))
        D = inp.ndim - 1
        output = random.normal(key, (6, 12, 14, 27))
        N_features = [inp.shape[0], 33, output.shape[0]]
        kernel_size = 3
        N_conv = 3
        N_layers = 4

        N_modes = [3,]*D
        weights = [(-1)**jnp.arange(n) for n in inp.shape[1:]]
        '''
        keys = random.split(key, 4)

        self.encoder = DilatedConvBlock([N_features[0], N_features[1]], [[1]*D, ], [[1]*D, ], keys[0], activation=relu)
        self.decoder = DilatedConvBlock([N_features[1], N_features[2]], [[1]*D, ], [[1]*D, ], keys[1], activation=relu)

        keys = random.split(keys[2], N_layers)
        self.processor = [DilatedConvBlock([N_features[1],]*(N_conv + 1), [[1,]*D, ]*N_conv, [[kernel_size,]*D, ]*N_conv, k, activation=relu) for k in keys]

        keys = random.split(keys[3], N_layers)
        self.conv = [eqx.nn.Conv(num_spatial_dims=D, in_channels=N_features[1], out_channels=N_features[1], kernel_size=[1,]*D, key=key) for key in keys]
        self.axes = [i+1 for i in range(D)]
        self.weights = weights
        self.N_pad = [[0, 0],] + [[0, w.shape[0] - m] for m, w in zip(N_modes, weights)]
        self.N_cut = [[0,]*(D+1), [N_features[1],] + N_modes]

    def __call__(self, x):
        x = self.encoder(x)
        for i, (pr, conv) in enumerate(zip(self.processor, self.conv)):
            y = conv(x)
            x = cheb.values_to_coefficients(x, self.axes, self.weights)
            x = slice(x, *self.N_cut)
            x = pr(x)
            x = jnp.pad(x, self.N_pad)
            x = cheb.coefficients_to_values(x, self.axes, self.weights)
            if i < (len(self.processor) - 1):
                x = relu(x + y)
            else:
                x = x + y
        x = self.decoder(x)
        return x

def compute_loss(model, input, target):
    output = vmap(model)(input)
    l = jnp.mean(jnp.linalg.norm((output - target).reshape(input.shape[0], -1,), axis=1))
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, input, target, optim, opt_state):
    loss, grads = compute_loss_and_grads(model, input, target)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
