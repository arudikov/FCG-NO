import jax.numpy as jnp
from jax.scipy.linalg import toeplitz

from jax.lax import scan, dot_general
from jax import config
config.update("jax_enable_x64", True)

def DCT_I_a(input, axis):
    s = tuple([slice(None),]*axis + [slice(-2, 0, -1)])
    input = jnp.concatenate([input, input[s]], axis=axis)
    input = jnp.real(jnp.fft.rfft(input, axis=axis))
    return input

def DCT_I(input, axes):
    for a in axes:
        input = DCT_I_a(input, a)
    return input

def values_to_coefficients_a(input, axis, w):
    # for axis with size N, weights should be w = (-1)**jnp.arange(N)
    slice_ = tuple([slice(None),]*axis + [slice(-2, 0, -1)])
    skip = [slice(None),]*axis
    s_w = [1,]*axis + [-1,] + [1,]*(input.ndim-axis-1)
    input = jnp.concatenate([input, input[slice_]], axis=axis)
    input = jnp.real(jnp.fft.rfft(input, axis=axis)) * (2 / input.shape[axis])

    input = input.at[tuple(skip + [0])].set(input[tuple(skip + [0])] / 2)
    input = input.at[tuple(skip + [-1])].set(input[tuple(skip + [-1])] / 2)

    input = input * w.reshape(s_w)
    return input

def values_to_coefficients(input, axes, weights):
    for i in range(len(axes)):
        input = values_to_coefficients_a(input, axes[i], weights[i])
    return input

def coefficients_to_values_a(input, axis, w):
    # for axis with size N, weights should be w = (-1)**jnp.arange(N)
    slice_ = tuple([slice(None),]*axis + [slice(-2, 0, -1)])
    skip = [slice(None),]*axis
    s_w = [1,]*axis + [-1,] + [1,]*(input.ndim-axis-1)

    input = input * w.reshape(s_w)

    input = input.at[tuple(skip + [-1])].set(input[tuple(skip + [-1])] * 2)
    input = input.at[tuple(skip + [0])].set(input[tuple(skip + [0])] * 2)

    input = jnp.concatenate([input, input[slice_]], axis=axis)
    input = jnp.real(jnp.fft.rfft(input, axis=axis)) / 2
    return input

def coefficients_to_values(input, axes, weights):
    for i in range(len(axes)):
        input = coefficients_to_values_a(input, axes[i], weights[i])
    return input

def get_cheb_grid(N):
    n = jnp.arange(N)
    x = -jnp.cos(n*jnp.pi/(N-1))
    return x

def get_transform_weights(l, N):
    if l > 0:
        r = jnp.arange(N)
        weights = l / (l + r)
    else:
        weights = jnp.hstack([jnp.ones((1,)), jnp.ones(N-1)*0.5])
    return weights

def ultraspherical_transform(u, weights, axis):
    skip = [slice(None),]*axis
    s_w = [1,]*axis + [-1,] + [1,]*(u.ndim-axis-1)
    u_ = u*weights.reshape(s_w)
    u_ = u_.at[tuple(skip + [slice(None, -2)])].set(u_[tuple(skip + [slice(None, -2)])] - u[tuple(skip + [slice(2, None)])]*weights.reshape(s_w)[tuple(skip + [slice(2, None)])])
    return u_

def get_diff_weights(l, N):
    return jnp.arange(N-l) + l

def diff(u, l, weights, axis):
    skip = [slice(None),]*axis
    s_w = [1,]*axis + [-1,] + [1,]*(u.ndim-axis-1)
    u_ = jnp.zeros_like(u)
    u_ = 2**(l-1)*jnp.prod(jnp.arange(1, l))*u_.at[tuple(skip + [slice(None, -l),])].set(weights.reshape(s_w)*u[tuple(skip + [slice(l, None),])])
    return u_

def get_derivative_data(l, N):
    transform_weights = [get_transform_weights(l_, N) for l_ in range(l)]
    diff_weights = get_diff_weights(l, N)
    return transform_weights, diff_weights

def ultraspherical_chain_transform(u, weights, axis):
    for w in weights:
        u = ultraspherical_transform(u, w, axis)
    return u

def hankel_(carry, i):
    carry = jnp.roll(carry, -1)
    carry = carry.at[-1].set(0)
    return carry, carry

def hankel(c):
    _, A = scan(hankel_, c, jnp.arange(c.shape[-1]-1))
    return jnp.concatenate([c.reshape(1, -1), A], 0)

def get_multiplication_matrix(c, M):
    # f1 = \sum c T, f2 = \sum d U, f1*f2 = \sum e U => e = A(c)d
    T = jnp.eye(M, k=0)*c[0] / 2 + toeplitz(jnp.pad(c, (0, M-c.shape[0]))) / 2 - hankel(jnp.pad(c[2:], (0, M-c.shape[0] + 2))) / 2
    return T

def get_interpolation_matrix(evaluate_at, m):
    known_at = get_cheb_grid(m)
    points = jnp.arange(m)[::-1]
    weights = jnp.array(((-1)**points)*jnp.ones(m).at[0].set(1/2).at[-1].set(1/2))
    n = evaluate_at.shape[0]
    W = jnp.dot(jnp.ones((n, 1)), weights.reshape(1, m))/(jnp.dot(evaluate_at.reshape(-1, 1), jnp.ones((1, m))) - jnp.dot(jnp.ones((n, 1)), known_at.reshape(1, m)))
    mask = jnp.isinf(W)
    marked_rows = jnp.logical_not(jnp.sum(mask, axis=1)).reshape((-1, 1))
    W = jnp.nan_to_num(W*marked_rows, nan=1.0)
    return W

def get_interpolation_data(grids, M):
    W = []
    for g, m in zip(grids, M):
        w = get_interpolation_matrix(g, m)
        w = w / jnp.sum(w, 1).reshape(-1, 1)
        W.append(w)
    return W

def interp(f, W):
    '''
    Example of usage:
    M = 100
    N = 70
    y = get_cheb_grid(N)
    x = 2*jnp.linspace(0, 1, M) - 1

    X, Y = jnp.meshgrid(y, y)
    f = jnp.expand_dims(jnp.cos(jnp.pi*(X+4*Y))*Y**2, 0)

    W = get_interpolation_data([x, x], [N, N])
    f_interp = interp(f, W)
    '''
    for w in W[::-1]:
        f = dot_general(w, f, (((1,), (f.ndim-1,)), ((), ())))
    f = jnp.moveaxis(f, f.ndim-1, 0)
    return f
