import jax.numpy as jnp
import equinox as eqx

from jax import config, random, vmap
from jax.nn import relu
from typing import Callable
config.update("jax_enable_x64", True)


class ConvNet(eqx.Module):
    convs: list
#     activation: Callable

    def __init__(self, D, features, odd_kernel, key, N_convs):
        keys = random.split(key, N_convs)
#         self.activation = relu
        self.convs = [eqx.nn.Conv(D, features, features, odd_kernel, padding=(odd_kernel - 1) // 2, key=key) for key in keys[:-1]]

    def __call__(self, x):
        for c in self.convs:
            x = relu(c(x))
        return x

class UNet(eqx.Module):
    transofrm_convs: list
    before_convs: list
    after_d_convs: list
    after_convs: list
    right_after_convs: list
    pool: list
    conv_t: list

    def __init__(self, D, staring_N, features, kernel_size, N_convs, key, depth=5):
        input_features, internal_features, output_features = features
        features_layers = [internal_features*2**i for i in range(depth)]
        even_kernel = 2 * (kernel_size // 2)
        odd_kernel = 2 * (kernel_size // 2) + 1
        N_down = [staring_N//2**i for i in range(depth)]
        upsampling_kernels = []
        for i in range(depth-1):
            if 2*N_down[::-1][i] == N_down[::-1][i+1]:
                upsampling_kernels.append(even_kernel)
            else:
                upsampling_kernels.append(odd_kernel)

        # for the transformation of input and output
        keys = random.split(key, 3)
        self.transofrm_convs = [eqx.nn.Conv(D, input_features, internal_features, odd_kernel, padding=(odd_kernel - 1) // 2, key=keys[0])]
        self.transofrm_convs += [eqx.nn.Conv(D, internal_features, output_features, odd_kernel, padding=(odd_kernel - 1) // 2, key=keys[1])]

        # convolutions before downsampling
        keys = random.split(keys[-1], depth+1)
        self.before_convs = [ConvNet(D, features, odd_kernel, key, N_convs) for features, key in zip(features_layers, keys[:-1])]

        # convolutions after downsampling
        keys = random.split(keys[-1], depth)
        self.after_d_convs = [eqx.nn.Conv(D, feature, 2*feature, odd_kernel, padding=(odd_kernel - 1) // 2, key=key) for feature, key in zip(features_layers[:-1], keys[:-1])]

        # convolutions after upsampling
        keys = random.split(keys[-1], depth+2)
        self.after_convs = [ConvNet(D, features, odd_kernel, key, N_convs) for features, key in zip(features_layers, keys[:-1])][::-1]

        # transition conv right after upsampling
        keys = random.split(keys[-1], depth)
        self.right_after_convs = [eqx.nn.Conv(D, 2*feature, feature, odd_kernel, padding=(odd_kernel - 1) // 2, key=key) for feature, key in zip(features_layers[:-1], keys[:-1])][::-1]

        # pooling
        self.pool = [eqx.nn.AdaptivePool(N_down[i+1], D, jnp.max) for i in range(depth-1)]

        # upsampling
        keys = random.split(keys[-1], depth-1)
        self.conv_t = [eqx.nn.ConvTranspose(D, f_in, f_out, kernel_size, padding=(kernel_size - 2) // 2, stride=2, key=key) for f_in, f_out, kernel_size, key in zip(features_layers[::-1][:-1], features_layers[::-1][1:], upsampling_kernels, keys)]

    def __call__(self, x):
        X = [self.before_convs[0](self.transofrm_convs[0](x))]
        for i, p in enumerate(self.pool):
            x_ = self.before_convs[i+1](self.after_d_convs[i](p(X[-1])))
            X.append(x_)

        x_ = self.after_convs[0](X[-1])

        for i, up in enumerate(self.conv_t):
            x_ = up(x_)
            x_ = self.after_convs[i+1](self.right_after_convs[i](jnp.vstack([X[len(X)-2-i], x_])))

        x_ = self.transofrm_convs[1](x_)
        return x_

def compute_loss(model, input, target):
    output = vmap(lambda z: model(z), in_axes=(0,))(input)
    l = jnp.mean(jnp.linalg.norm((output - target).reshape(input.shape[0], -1), axis=1)**2)
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, input, target, optim, opt_state):
    loss, grads = compute_loss_and_grads(model, input, target)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_jit
def make_step_m(model, input, target, optim, opt_state):
    # for optimizers that require model for update, e.g. adamw
    loss, grads = compute_loss_and_grads(model, input, target)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
