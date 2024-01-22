# +
import jax.numpy as jnp

from transforms import utilities as ut
from transforms import library_of_transforms as lft
from jax import config, random, vmap
config.update("jax_enable_x64", True)


# +
def polynomial_interpolation(data, poly_in, parameters_in, grids_out):
    '''
    len(data.shape) - 1 = len(poly_in) = len(grids_out) = len(parameters_in)
    Example input:
    data.shape = (batch, features, x, t)
    poly_in = ["poly_x", "poly_t"]
    parameters_in = [[1, 1], [2.3, 14.2]]
    grids_out = [jnp.array, jnp.array] -- do not forget that grid should be transformed, i.e., points should lay where polynomials are defined
    '''
    data_c = ut.transform_data(data, poly_in, parameters_in, "analysis", grids_out)
    return ut.transform_data(data_c, poly_in, parameters_in, "synthesis", grids_out)

def linear_interpolation(data, grids_in, grids_out):
    '''
    len(data.shape) - 1 = len(grids_in) = len(grids_out)
    Example input:
    data.shape = (batch, features, x, t)
    grids_in = [jnp.array, jnp.array]
    grids_out = [jnp.array, jnp.array]
    '''
    s = data.shape
    if len(s) == 3:
        data = vmap(jnp.interp, in_axes=(None, None, 0))(grids_out[0], grids_in[0], data.reshape(-1, s[-1]))
        return data.reshape(s[0], s[1], len(grids_out[0]))
    elif len(s) == 4:
        data = data.reshape(-1, s[2], s[3])
        data = vmap(vmap(jnp.interp, in_axes=(None, None, 0)), in_axes=(None, None, 0))(grids_out[1], grids_in[1], data)
        data = vmap(vmap(jnp.interp, in_axes=(None, None, 0)), in_axes=(None, None, 2))(grids_out[0], grids_in[0], data)
        data = jnp.transpose(data, [1, 2, 0])
        return data.reshape(s[0], s[1], len(grids_out[0]), len(grids_out[1]))
    else:
        return None


# -

def get_coeff(data, poly_in, parameters_in):
  # data.shape = (batch, features, x1, x2, ...) computed on grids x1, x2, ... coresponding to polynomials in poly_in
  return ut.transform_data(data, poly_in, parameters_in, "analysis", data.shape[2:])
