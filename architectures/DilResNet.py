import jax.numpy as jnp
import equinox as eqx

from typing import Callable
from jax import config, random, vmap
from jax.nn import relu, gelu
config.update("jax_enable_x64", True)


class DilatedConvBlock(eqx.Module):
    convolutions: list
#     activation: Callable

    def __init__(self, channels, dilations_D, kernel_sizes_D, key, activation=gelu):
        # 1D example: channels = [3, 40, 40, 1], dilations_D = [[1,], [1,], [1]], kernel_sizes_D = [[3,], [3,], [3,]]
        # 2D example: channels = [3, 40, 40, 1], dilations_D = [[4, 4], [3, 3], [2, 1]], kernel_sizes_D = [[3, 4], [3, 4], [4, 5]]
        kernel_sizes_D = [[k if k%2 == 1 else k+1 for k in kernel_sizes] for kernel_sizes in kernel_sizes_D]
        paddings_D = [[d*(k//2) for d, k in zip(dilations, kernel_sizes)] for dilations, kernel_sizes in zip(dilations_D, kernel_sizes_D)]
        keys = random.split(key, len(channels))
        D = len(kernel_sizes_D[0])
        self.convolutions = [eqx.nn.Conv(num_spatial_dims=D, in_channels=f_i, out_channels=f_o, dilation=d, kernel_size=k, padding=p, key=key) for f_i, f_o, d, k, p, key in zip(channels[:-1], channels[1:], dilations_D, kernel_sizes_D, paddings_D, keys)]
#         self.activation = activation

    def __call__(self, x):
        for conv in self.convolutions[:-1]:
            x = gelu(conv(x))
        x = self.convolutions[-1](x)
        return x

    def linear_call(self, x):
        for conv in self.convolutions:
            x = conv(x)
        return x

class DilatedResNet(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    processor: list
#     activation: Callable

    def __init__(self, key, channels, n_cells, activation=relu, kernel_size=3, D=1):
        in_channels, processor_channels, out_channels = channels
        keys = random.split(key, 3)
        self.encoder = DilatedConvBlock([in_channels, processor_channels], [[1,]*D,], [[kernel_size,]*D,], keys[0])
        self.decoder = DilatedConvBlock([processor_channels, out_channels], [[1,]*D,], [[kernel_size,]*D,], keys[1])
        keys = random.split(keys[2], n_cells)
        channels_ = [processor_channels,]*8
        dilations = [[1,]*D, [2,]*D, [4,]*D, [8,]*D, [4,]*D, [2,]*D, [1,]*D]
        kernel_sizes = [[kernel_size,]*D,]*7
        self.processor = [DilatedConvBlock(channels_, dilations, kernel_sizes, key, activation=activation) for key in keys]
#         self.activation = activation

    def __call__(self, x):
        x = self.encoder(x)
        for pr in self.processor[:-1]:
            x = gelu(pr(x)) + x
        x = self.processor[-1](x) + x
        x = self.decoder(x)
        return x

    def linear_call(self, x):
        x = self.encoder(x)
        for pr in self.processor:
            x = pr(x) + x
        x = self.decoder(x)
        return x

class DilatedResNet_truncated(DilatedResNet):
    pooling: eqx.Module

    def __init__(self, key, channels, n_cells, output_shape, activation=relu, kernel_size=3, D=1):
        super().__init__(key, channels, n_cells, activation=activation, kernel_size=kernel_size, D=D)
        self.pooling = eqx.nn.AdaptivePool(output_shape[1:], D, jnp.mean)

    def __call__(self, x):
        x = super().__call__(x)
        x = self.pooling(x)
        return x

class pooledDilatedResNet(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    processor: list
    pooling: list
#     activation: Callable

    def __init__(self, key, channels, n_cells, input_shape, activation=relu, pooling_operation=jnp.mean, reduction_factor=2, kernel_size=3, D=1):
        in_channels, processor_channels, out_channels = channels
        keys = random.split(key, 3)
        operation = lambda x: jnp.mean(x)
        self.encoder = DilatedConvBlock([in_channels, processor_channels], [[1,]*D,], [[kernel_size,]*D,], keys[0])
        self.decoder = DilatedConvBlock([processor_channels, out_channels], [[1,]*D,], [[kernel_size,]*D,], keys[1])
        keys = random.split(keys[2], n_cells)
        channels_ = [processor_channels,]*8
        dilations = [[1,]*D, [2,]*D, [4,]*D, [8,]*D, [4,]*D, [2,]*D, [1,]*D]
        kernel_sizes = [[kernel_size,]*D,]*7
        self.processor = [DilatedConvBlock(channels_, dilations, kernel_sizes, key, activation=activation) for key in keys]
#         self.activation = activation

        target_shapes = [tuple([max(int(s // (reduction_factor*(i+1))), 1) for s in input_shape[1:]]) for i in range(n_cells)]
        self.pooling = [eqx.nn.AdaptivePool(s, num_spatial_dims=D, operation=pooling_operation) for s in target_shapes]

    def __call__(self, x):
        x = self.encoder(x)
        for pr, pool in zip(self.processor, self.pooling):
            x = pool(gelu(pr(x)) + x)
        x = self.decoder(x)
        return x

class ResNet(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    processor: list
#     activation: Callable

    def __init__(self, key, channels, conv_per_cell, n_cells, activation=relu, kernel_size=3, D=1):
        in_channels, processor_channels, out_channels = channels
        keys = random.split(key, 3)
        self.encoder = DilatedConvBlock([in_channels, processor_channels], [[1,]*D,], [[kernel_size,]*D,], keys[0])
        self.decoder = DilatedConvBlock([processor_channels, out_channels], [[1,]*D,], [[kernel_size,]*D,], keys[1])
        keys = random.split(keys[2], n_cells)
        channels_ = [processor_channels,]*conv_per_cell
        dilations = [[1,]*D, ]*conv_per_cell
        kernel_sizes = [[kernel_size,]*D,]*conv_per_cell
        self.processor = [DilatedConvBlock(channels_, dilations, kernel_sizes, key, activation=activation) for key in keys]
#         self.activation = activation

    def __call__(self, x):
        x = self.encoder(x)
        for pr in self.processor:
            x = gelu(pr(x)) + x
        x = self.decoder(x)
        return x

class pooledResNet(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    processor: list
    pooling: list
#     activation: Callable

    def __init__(self, key, channels, conv_per_cell, n_cells, input_shape, activation=gelu, reduction_factor=2, pooling_operation=jnp.mean, kernel_size=3, D=1):
        in_channels, processor_channels, out_channels = channels
        keys = random.split(key, 3)
        self.encoder = DilatedConvBlock([in_channels, processor_channels], [[1,]*D,], [[kernel_size,]*D,], keys[0])
        self.decoder = DilatedConvBlock([processor_channels, out_channels], [[1,]*D,], [[kernel_size,]*D,], keys[1])
        keys = random.split(keys[2], n_cells)
        channels_ = [processor_channels,]*conv_per_cell
        dilations = [[1,]*D, ]*conv_per_cell
        kernel_sizes = [[kernel_size,]*D,]*conv_per_cell
        self.processor = [DilatedConvBlock(channels_, dilations, kernel_sizes, key, activation=activation) for key in keys]
#         self.activation = activation

        target_shapes = [tuple([max(int(s // (reduction_factor*(i+1))), 1) for s in input_shape[1:]]) for i in range(n_cells)]
        self.pooling = [eqx.nn.AdaptivePool(s, num_spatial_dims=D, operation=pooling_operation) for s in target_shapes]

    def __call__(self, x):
        x = self.encoder(x)
        for pr, pool in zip(self.processor, self.pooling):
            x = pool(gelu(pr(x)) + x)
        x = self.decoder(x)
        return x

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
