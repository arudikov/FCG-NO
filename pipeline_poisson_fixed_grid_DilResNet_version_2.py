import warnings
warnings.filterwarnings('ignore')

import os
import jax
import sys
import optax
import cloudpickle
import itertools
import equinox as eqx
import jax.numpy as jnp
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import argparse

from transforms.interpolation import linear_interpolation
from transforms import library_of_transforms as lft
from solvers import FD_2D
from tqdm import tqdm
from jax import device_put, clear_caches
from functools import partial
from jax.scipy.sparse.linalg import cg
from transforms import utilities, cheb
from collections import namedtuple
from jax.lax import scan, dot_general
from architectures import DilResNet
from jax import config, random, grad, vmap, pmap, tree_map, tree_leaves, jit
from jax.experimental import sparse as jsparse
from scipy.interpolate import interpn

config.update("jax_enable_x64", True)

@jit
def spsolve_scan(carry, n):
    A, r = carry
    A_bcsr = jsparse.BCSR.from_bcoo(A)
    Ar = jsparse.linalg.spsolve(A_bcsr.data[n], A_bcsr.indices[n], A_bcsr.indptr[n], r[n])
    return [A, r], Ar

def res_func(A, B, res):
    _, Ar = scan(spsolve_scan, [A, res], jnp.arange(A.shape[0]))
    Ar = jnp.array(Ar)
    B_Ar = jsparse.bcoo_dot_general(A, B - Ar, dimension_numbers=((2, 1), (0, 0)))
    numerator = jnp.sqrt(jnp.einsum('bi, bi -> b', B - Ar, B_Ar))
    denominator = jnp.sqrt(jnp.einsum('bi, bi -> b', Ar, res))
    value = numerator / denominator
    return value

def random_polynomial_2D(x, y, coeff):
    res = 0
    for i, j in itertools.product(range(coeff.shape[0]), repeat=2):
        res += coeff[i, j]*jnp.exp(2*jnp.pi*x*i*1j)*jnp.exp(2*jnp.pi*y*j*1j)/(1+i+j)**2
    res = jnp.real(res)
    return res

def get_functions(key):
    c_ = random.normal(key, (1, 5, 5), dtype=jnp.complex128)
    rhs = lambda x, y, c=c_[0]: random_polynomial_2D(x, y, c)
    return rhs

def dataset(grid, N_samples, key):
    keys = random.split(key, N_samples)
    A, rhs = [], []
    
    for key in keys:
        rhs_sample, A_sample = FD_2D(grid, [lambda x, y: 1, get_functions(key)])
        A.append(A_sample.reshape(1, grid**2, -1))
        rhs.append(rhs_sample)
    A = device_put(jsparse.bcoo_concatenate(A, dimension=0))
    return A, jnp.array(rhs)

def get_exact_solution(A, rhs, grid, N_samples):
    A_bcsr = jsparse.BCSR.from_bcoo(A)
    u_exact = jnp.stack([jsparse.linalg.spsolve(A_bcsr.data[n], A_bcsr.indices[n], A_bcsr.indptr[n], rhs[n].reshape(-1,)) for n in range(N_samples)]).reshape(N_samples, grid, grid)
    return u_exact

def get_DilResNet(key, features_train, grid):
    input = features_train
    D = len(input.shape[1:])
    
    input_shape = input.shape[0]
    kernel_size = 3
    n_cells = 4
    N_features_out = 1
    
    if grid <= 64:
        N_features = 24
    else:
        N_features = grid // 8
        
    channels = [input_shape, N_features, N_features_out]
    model = DilResNet.DilatedResNet(key, channels, n_cells, kernel_size=kernel_size, D=D)
    
    # Parameters of training
    N_epoch = 150
    if grid <= 64:
        batch_size = 32
    else:
        batch_size = 1024 // grid

    if grid < 128:
        learning_rate = 1e-3
    else:
        learning_rate = 8e-4
    
    model_data = {
        "model_name": "DilResNet",
        "model": model
        }
    
    optimization_specification = {
        "learning_rate": learning_rate,
        "compute_loss": lambda carry, indices: compute_loss_scan(carry, indices),
        "make_step": lambda carry, indices, optim: make_step_scan(carry, indices, optim),
        "N_epochs": N_epoch,
        "batch_size": batch_size, 
        "res_func": lambda A, model, input: res_func(A, model, input) 
    }
    return model_data, optimization_specification

def FCG(A, features, model, N_iter, m_max, optimization_specification, eps=1e-30, count_values=False, j=0):
    samples = features.shape[0]
    n = features.shape[-1]

    X = jnp.zeros((samples, n, N_iter+1))
    R = jnp.zeros((samples, n, N_iter+1))
    P_list, S_list = [], []

    X = X.at[:, :, 0].set(random.normal(random.PRNGKey(j), (samples, n)))
    R = R.at[:, :, 0].set(features - jsparse.bcoo_dot_general(A, X[:, :, 0], dimension_numbers=((2, 1),(0, 0))))

    grid = int(n**0.5)
    h = 1. / grid

    values = []
    times = []

    for idx in range(N_iter):
        start = time.time()
        norm = jnp.linalg.norm(R[:, :, idx], axis=1)
        
        train_data = [jnp.einsum('bi,b->bi', R[:, :, idx], 1./norm)]
        if type(model) != type(lambda x: x):
            output = vmap(model, in_axes=(0))(jnp.einsum('bi, b->bi', R[:, :, idx], 1/norm)[:, None].reshape(-1, 1, grid, grid))[:, 0].reshape(-1, grid**2)
            U = jnp.einsum('bi, b->bi', output, norm)

        else:
            history_train = []
            U = vmap(model)(R[:, :, idx])
        if count_values:
            value = optimization_specification['res_func'](A,  U, R[:, :, idx])
            values.append(value)

        P = U
        for k in range(len(P_list)):
            alpha = - jnp.einsum('bj, bj->b', S_list[k], U) / (jnp.einsum('bj, bj->b', S_list[k], P_list[k]) + eps)
            P += jnp.einsum('b, bj->bj', alpha, P_list[k])
        
        S = jsparse.bcoo_dot_general(A, P, dimension_numbers=((2, 1),(0, 0)))
        beta = jnp.einsum('bj, bj -> b', P, R[:, :, idx]) / (jnp.einsum('bj, bj -> b', S, P) + eps)

        X = X.at[:, :, idx+1].set(X[:, :, idx] + jnp.einsum('b, bj->bj', beta, P))
        R = R.at[:, :, idx+1].set(R[:, :, idx] - jnp.einsum('b, bj->bj', beta, S))
        
        if (idx % (m_max+1) == m_max) | (idx % (m_max + 1) == 0):
            P_list = []
            S_list = []
            
        P_list.append(P)
        S_list.append(S)
        end = time.time()
        times.append(end - start)
        
    return P, R, X, values

@jit
def compute_loss_scan(carry, indices):
    model, A, x, error, N_repeats = carry
    A, x, error = A[indices // N_repeats], x[indices], error[indices]
    B = vmap(lambda z: model(z), in_axes=(0,))(x[:, None, :])[:, 0].reshape(x.shape[0], -1)
    B_e = jsparse.bcoo_dot_general(A, B - error, dimension_numbers=((2, 1), (0, 0)))
    A_e = jsparse.bcoo_dot_general(A, error, dimension_numbers=((2, 1), (0, 0)))
    return carry, jnp.mean(jnp.sqrt(jnp.einsum('bi, bi -> b', B - error, B_e) / jnp.einsum('bi, bi -> b', error, A_e)))

# Notay loss
def compute_loss(model, A, x, error):
    B = vmap(lambda z: model(z), in_axes=(0,))(x[:, None, :])[:, 0].reshape(x.shape[0], -1)
    B_e = jsparse.bcoo_dot_general(A, B - error, dimension_numbers=((2, 1), (0, 0)))
    A_e = jsparse.bcoo_dot_general(A, error, dimension_numbers=((2, 1), (0, 0)))
    return jnp.mean(jnp.sqrt(jnp.einsum('bi, bi -> b', B - error, B_e) / jnp.einsum('bi, bi -> b', error, A_e)))

# l2-loss
# def compute_loss(model, A, input, target):
#     output = vmap(lambda z: model(z), in_axes=(0,))(input[:, None, :]).reshape(input.shape[0], -1)
#     l = jnp.mean(jnp.linalg.norm((output - target), axis=1)**2)
#     return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step_scan(carry, indices, optim):
    model, A, r, error, opt_state, N_repeats = carry
    loss, grads = compute_loss_and_grads(model, A[indices // N_repeats], r[indices], error[indices])
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, A, r, error, opt_state, N_repeats], loss

def train_on_epoch(key, batch_size, A, model, x, error, opt_state, make_step, N_repeats):
    N_samples = len(x)
    list_of_indices = jnp.linspace(0, N_samples-1, N_samples, dtype=jnp.int64)

    n_batches = N_samples // batch_size
    
    carry = [model, A, x, error, opt_state, N_repeats]
    n = random.choice(key, list_of_indices, shape = (n_batches, batch_size))
    data, epoch_loss = scan(make_step, carry, n)
    model = data[0]
    opt_state = data[4]
    return epoch_loss, model, opt_state

def test_on_epoch(key, batch_size, A, model, x, error, compute_loss, N_repeats):
    N_samples = len(x)
    list_of_indices = jnp.linspace(0, N_samples-1, N_samples, dtype=jnp.int64)

    n_batches = N_samples // batch_size
    
    n = random.choice(key, list_of_indices, shape = (n_batches, batch_size))
    carry = [model, A, x, error, N_repeats]
    data, epoch_loss = scan(compute_loss, carry, n)
    return epoch_loss

def train_model(model, A, x, error, optimization_specification, N_repeats):
    model = model
    history = []
    history_test = []

    c = x.shape[0] // optimization_specification['batch_size']
    keys = [value * c for value in np.arange(50, 1000, 50)]
    values = [0.5, ] * len(keys)
    dict_lr = dict(zip(keys, values))

    sc = optax.piecewise_constant_schedule(optimization_specification['learning_rate'], dict_lr)
    optimizer = optax.adamw(sc, weight_decay=1e-2)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    make_step = lambda a, b: optimization_specification['make_step'](a, b, optimizer)
    
    for it in tqdm(range(optimization_specification['N_epochs'])):
        key = random.PRNGKey(it)
        loss, model, opt_state = train_on_epoch(key, optimization_specification['batch_size'], A, model, x, error, opt_state, make_step, N_repeats)
        history.append(loss)
    return model, history

def get_dataset_rhs(state, key, grid):
    rhs = state[0]
    rhs = get_functions(key)
    
    x = jnp.linspace(0, 1, num=grid+2)[1:-1]
    y = jnp.linspace(0, 1, num=grid+2)[1:-1]

    xx, yy = jnp.meshgrid(x, y)
    
    state = [rhs(xx, yy)]
    return state, state

def generate_res_errors(grid, N_samples, N_repeats, A):
    residuals = [] 
    error = []
    for i in range(N_repeats): 
        keys = random.split(random.PRNGKey(i), 2)
        X_random = random.normal(keys[0], (N_samples, grid**2))

        initial = [jnp.zeros(shape=(grid, grid))]
        get_dataset_rhs_ = lambda x, k:  get_dataset_rhs(x, k, grid)
        rhs_random = scan(get_dataset_rhs_, initial, random.split(keys[1], N_samples))[1][0].reshape(-1, grid**2)

        residuals_random = rhs_random - jsparse.bcoo_dot_general(A, X_random, dimension_numbers=((2, 1),(0, 0)))
        norm = jnp.linalg.norm(residuals_random, axis=1)
        residuals_random = jnp.einsum('bij, b -> bij', residuals_random.reshape(-1, grid, grid), 1./norm)

        u_exact = get_exact_solution(A, rhs_random, grid=grid, N_samples=N_samples)

        error_random = u_exact - X_random.reshape((N_samples, -1, grid))
        error_random = jnp.einsum('bij, b -> bij', error_random, 1./norm).reshape(N_samples, -1)
        residuals.append(residuals_random)
        error.append(error_random)

    residuals = jnp.concatenate(residuals, axis=0)
    error = jnp.concatenate(error, axis=0)
    return residuals, error

def save_data(model, values, R, history, path, experiment_name):
    data = {
        "loss_model": values,
        "R_model": R,
        "history": history, 
    }
      
    with open(path + f"{experiment_name}_data.npz", "wb") as f:
        jnp.savez(f, **data)
    if type(model) != type(lambda x: x):
        with open(path + f"{experiment_name}_model", "wb") as f:
            cloudpickle.dump(model, f)

def main(model_type, train_generation, grid, samples_div, N_repeats, m_max, path):
    h = 1. / grid
    if model_type == 'DilResNet':
        key = random.PRNGKey(2)
        N_samples = grid//samples_div
        A_train, rhs_train = dataset(grid=grid, N_samples=N_samples, key=key)
        if train_generation == 'FCG':
            model_ = lambda x: x
            optimization_specification = {"res_func": lambda A, B, input: res_func(A, B, input)}
            R_, X_ = [], []
            for j in tqdm(range(N_samples // 2)):
                _, R, X, _ = FCG(A_train[j*2:(j+1)*2], rhs_train[j*2:(j+1)*2], model=model_, N_iter=N_repeats-1, m_max=m_max, optimization_specification=optimization_specification, count_values=False, j=j)
                R_.append(device_put(R, device=jax.devices("cpu")[0]))
                X_.append(device_put(X, device=jax.devices("cpu")[0]))
                
            del R, X
            clear_caches()
            
            R_ = jnp.concatenate(R_, axis=0)
            X_ = jnp.concatenate(X_, axis=0)

            residuals = R_.transpose(0, 2, 1).reshape(-1, grid, grid)
            norm = jnp.linalg.norm(residuals, axis=(1, 2))
            
            residuals = jnp.einsum('bij, b -> bij', residuals, 1./norm)
            u_exact = device_put(get_exact_solution(A_train, rhs_train, grid=grid, N_samples=N_samples), device=jax.devices("cpu")[0])
            
            del rhs_train
            clear_caches()
            
            error = (jnp.repeat(u_exact.reshape(-1, grid**2)[:, :, None], N_repeats, axis=2) - X_).transpose(0, 2, 1).reshape((N_samples * N_repeats, -1, grid))
            error = jnp.einsum('bij, b -> bij', error, 1./norm).reshape(N_samples * N_repeats, -1)
            
            del R_, X_, u_exact
            clear_caches()   
        else:
            residuals, error = generate_res_errors(grid, N_samples, N_repeats, A_train)
            
        model_data, optimization_specification = get_DilResNet(random.PRNGKey(40), residuals[0][None, ...], grid)
        model, history = train_model(model_data['model'], A_train, device_put(residuals, device=jax.devices("gpu")[0]),
                                     device_put(error, device=jax.devices("gpu")[0]), optimization_specification, N_repeats)

        del residuals, error, A_train
        clear_caches()
        
        key = random.PRNGKey(12)
        N_samples = 20
        A_test, rhs_test = dataset(grid=grid, N_samples=N_samples, key=key)
        N_iter = min(int(grid*2.5), 400)
        
        R, values = [], []
        count_values = False
        for j in tqdm(range(N_samples//4)):
            _, R_, _, values_ = FCG(A_test[j*4:(j+1)*4], rhs_test[j*4:(j+1)*4], model=model, N_iter=N_iter, m_max=m_max, optimization_specification=optimization_specification, count_values=True, j=j)
            R.append(R_)
            values.append(jnp.array(values_))
        
        del R_, values_
        clear_caches()
        
        R = jnp.concatenate(R, axis=0)
        
        if count_values:
            values = jnp.concatenate(values, axis=1).mean(axis=1)
    
    if model_type == 'Id':
        key = random.PRNGKey(12)
        N_samples = 20
        A_test, rhs_test = dataset(grid=grid, N_samples=N_samples, key=key)
        N_iter = min(grid*4, 400)
        
        model = lambda x: x
        history = []
        optimization_specification = {"res_func": lambda A, B, input: res_func(A, B, input)}
        
        R, values = [], []
        count_values = False
        for j in tqdm(range(N_samples//4)):
            _, R_, _, values_ = FCG(A_test[j*4:(j+1)*4], rhs_test[j*4:(j+1)*4], model=model, N_iter=N_iter, m_max=m_max, optimization_specification=optimization_specification, count_values=count_values, j=j)
            R.append(R_)
            values.append(jnp.array(values_))
        
        del R_, values_
        clear_caches()
        
        R = jnp.concatenate(R, axis=0)
        
        if count_values:
            values = jnp.concatenate(values, axis=1).mean(axis=1)
        
    save_data(model, values, R, history, path, f'{model_type}_{train_generation}_{grid}_{samples_div}_{N_repeats}_{m_max}')

    print(f'Done for {model_type}_{train_generation}_{grid}_{samples_div}_{N_repeats}_{m_max}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('--model', type=str, help='model type') # DilResNet, Id
    parser.add_argument('--gentype', type=str, help='generation type') # FCG, random
    parser.add_argument('--sdiv', type=int, help='sample divide') # N_samples=grid//sdiv
    parser.add_argument('--nrep', type=int, help='number of repeats')  # N_iter for FCG(I)
    parser.add_argument('--cuda', type=str, help='device cuda') # 
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    grids = [32]
    path = f'./Poisson/DilResNet/notay_loss_'
    # path = f'./Poisson/DilResNet/l2_loss_'
    m_max = 20
    
    for grid in grids:
        main(args.model, args.gentype, grid, args.sdiv, args.nrep, m_max, path)
