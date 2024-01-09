import jax.numpy as jnp
from functools import partial
from jax import jit, config
from jax.lax import dot_general

config.update("jax_enable_x64", True)

from transforms import library_of_transforms as lft

@jit
def d1(a, h):
    '''
    find derivative of a 1D functions given on uniform grid x

    a.shape = (N_features, N_x)
    h = grid spacing
    '''
    d_a = (jnp.roll(a, -1, axis=1) - jnp.roll(a, 1, axis=1)) / (2*h)
    d_a = d_a.at[:, 0].set((-3*a[:, 0]/2 + 2*a[:, 1] - a[:, 2]/2)/h) # 1/2	−2	3/2
    d_a = d_a.at[:, -1].set((a[:, -3]/2 - 2*a[:, -2] + 3*a[:, -1]/2)/h) # 1/2	−2	3/2
    return d_a

@jit
def d2(a, h):
    '''
    find second derivative of a 2D functions given on uniform grid x

    a.shape = (N_features, N_x)
    h = grid spacing
    '''
    [2, -5, 4, -1]
    d2_a = (- 2*a + jnp.roll(a, -1, axis=1) + jnp.roll(a, 1, axis=1)) / (h**2)
    d2_a = d2_a.at[:, 0].set((2*a[:, 0] - 5*a[:, 1] + 4*a[:, 2] - a[:, 3]) / h**2)
    d2_a = d2_a.at[:, -1].set((2*a[:, -1] - 5*a[:, -2] + 4*a[:, -3] - a[:, -4]) / h**2)
    return d2_a

@partial(jit, static_argnums=(2,))
def d_fd(a, h, axis):
    '''
    find first derivative of nD function given on uniform grid

    a.shape = (N_features, N_x, N_y, N_z, ...) - input array for taking derivative
    h = grid spacing
    axis = dimension to take a derivative, 1 corresponds to dx, 2 orresponds to dy, ...
    '''
    d_a = d1(jnp.moveaxis(a, axis, 1), h)
    return jnp.moveaxis(d_a, axis, 1)

@partial(jit, static_argnums=(2,))
def d2_fd(a, h, axis):
    '''
    find second derivative of nD function given on uniform grid

    a.shape = (N_features, N_x, N_y, N_z, ...) - input array for taking derivative
    h = grid spacing
    axis = dimension to take a derivative, 1 corresponds to dx, 2 orresponds to dy, ...
    '''
    d2_a = d2(jnp.moveaxis(a, axis, 1), h)
    return jnp.moveaxis(d2_a, axis, 1)

def get_diff_data(N, D=1, k=1):
    '''
    matrices needed for taking derivatives of the function evaluated at Gauss-Chebyshev grid nodes, i.e., lft.poly_data["Chebyshev_t"]["nodes"](N, [1, 1])

    N = number of nodes
    D = dimension of the array to be differentiated
    k = smoothing factor, only first N-k polynomials are taken into account
    '''
    x = lft.poly_data["Chebyshev_t"]["nodes"](N, [1, 1])
    A = lft.poly_data["Chebyshev_t"]["transform"](N, [1, 1])["analysis"]
    S = lft.poly_data["Chebyshev_u"]["transform"](N, [1, 1], grid=x)["synthesis"]
    n = jnp.arange(N) + 1
    n = n.at[-k:].set(0)
    for i in range(D):
        n = jnp.expand_dims(n, 1)
    return A, S, n

@jit
def d1_s(a, A, S, n):
    '''
    find derivative of a 1D functions given on grid lft.poly_data["Chebyshev_t"]["nodes"](N, [1, 1])

    a.shape = (N_features, N_x)
    A, S, n = array given by get_diff_data(N)
    '''
    d_a = dot_general(A, a, (((1,), (1,)), ((), ())))
    d_a = jnp.roll(d_a, -1, axis=0) * n
    d_a = dot_general(S, d_a, (((1,), (0,)), ((), ())))
    return jnp.moveaxis(d_a, 0, 1)

@partial(jit, static_argnums=(4,))
def d_s(a, A, S, n, axis):
    '''
    find first derivative of nD function given on tensor products of grids lft.poly_data["Chebyshev_t"]["nodes"](N, [1, 1])

    a.shape = (N_features, N_x, N_y, N_z, ...) - input array for taking derivative
    A, S, n = array given by get_diff_data(N)
    axis = dimension to take a derivative, 1 corresponds to dx, 2 orresponds to dy, ...
    '''
    d_a = d1_s(jnp.moveaxis(a, axis, 1), A, S, n)
    return jnp.moveaxis(d_a, axis, 1)

def get_pdiff_data(N, D=1):
    '''
    frequencies for periodic fucntions defined on grid lft.poly_data["Real_Fourier"]["nodes"](N, [1, 1])
    '''
    freq = 2*1j*jnp.pi*jnp.fft.fftfreq(N)*N
    freq = jnp.expand_dims(freq, 0)
    for i in range(D-1):
        freq = jnp.expand_dims(freq, 2)
    return freq

@jit
def d1_sp(a, freq):
    '''
    find derivative of a 1D periodic functions given on grid lft.poly_data["Real_Fourier"]["nodes"](N, [1, 1])

    a.shape = (N_features, N_x)
    freq = array given by get_pdiff_data(N_x)
    '''
    return jnp.real(jnp.fft.ifft(freq*jnp.fft.fft(a, axis=1), axis=1))

@jit
def d2_sp_(a, freq):
    '''
    find second derivative of a 1D periodic functions given on grid lft.poly_data["Real_Fourier"]["nodes"](N, [1, 1])

    a.shape = (N_features, N_x)
    A, S, n = array given by get_diff_data(N)
    '''
    return jnp.real(jnp.fft.ifft(freq**2*jnp.fft.fft(a, axis=1), axis=1))

@partial(jit, static_argnums=(2,))
def d_sp(a, freq, axis):
    d_a = d1_sp(jnp.moveaxis(a, axis, 1), freq)
    return jnp.moveaxis(d_a, axis, 1)

@partial(jit, static_argnums=(2,))
def d2_sp(a, freq, axis):
    d_a = d2_sp_(jnp.moveaxis(a, axis, 1), freq)
    return jnp.moveaxis(d_a, axis, 1)

@jit
def dn_sp_(a, freq, n):
    '''
    ns derivative of a 1D periodic functions given on grid lft.poly_data["Real_Fourier"]["nodes"](N, [1, 1])

    a.shape = (N_features, N_x)
    A, S, n = array given by get_diff_data(N)
    '''
    return jnp.real(jnp.fft.ifft(freq**n*jnp.fft.fft(a, axis=1), axis=1))

@partial(jit, static_argnums=(3,))
def dn_sp(a, freq, n, axis):
    d_a = dn_sp_(jnp.moveaxis(a, axis, 1), freq, n)
    return jnp.moveaxis(d_a, axis, 1)
