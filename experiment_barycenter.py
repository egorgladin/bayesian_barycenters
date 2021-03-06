"""
Experiment: find barycenter for a simple 3x3 setup.
"""
import numpy as np
import torch
import ot
import scipy.stats
import time
import gc

from algorithm import algorithm
from utils import safe_log, plot_trajectory, norm_sq, replace_zeros, get_sampler, get_cost_mat


def get_data_and_solution(device, dtype=torch.float32, size=3, column_interval=1):
    """
    Get two simple 3x3 images and their barycenter.

    :param device: 'cuda' or 'cpu'
    :return: list of (2, 9) tensor (data points) and (9,) tensor (barycenter)
    """
    cs = []
    for i in range(0, size, column_interval):
        if i == size // 2:
            pass
        else:
            c = torch.zeros((size, size), device=device, dtype=dtype)
            c[:, i] = 1. / size
            cs.append(c.flatten())

    barycenter = torch.zeros((size, size), device=device, dtype=dtype)
    barycenter[:, size // 2] = 1. / size

    cs = torch.stack(cs)
    return cs, barycenter.flatten()


def get_init_barycenter(r_opt, noise_level, seed=None):
    """Get initial point defined as the true barycenter + noise."""
    if seed is not None:
        torch.manual_seed(seed)
    zeros = torch.zeros_like(r_opt)
    noise = torch.normal(zeros, noise_level)
    r_prior = r_opt + torch.where(noise > 0., noise, zeros)
    r_prior /= r_prior.sum()

    # map to the whole Euclidean space
    z_prior = safe_log(r_prior)
    return r_prior, z_prior


def get_optimal_value(device, cost_mat, cs, r_opt):
    wasser_dist = [ot.emd2(c, r_opt, cost_mat) for c in cs]
    return sum(wasser_dist)


def experiment_pot(img_size, column_interval, n_steps, device, sample_size, prior_var, var_decay, noise_level=None, add_entropy=False,
                   start_coef=20., decrease=0.25, decay='exp', empir_cov=False, temperature=None, plot=False, track_time=False):
    dtype = torch.float64 if device == 'cpu' else torch.float32  # avoid errors when converting from torch to numpy
    cost_mat = get_cost_mat(img_size, device, dtype=dtype)
    cs, r_opt = get_data_and_solution(device, dtype=dtype, size=img_size, column_interval=column_interval)

    if device == 'cpu':
        cost_mat = cost_mat.numpy()
        cs = cs.numpy()
    else:
        cs = replace_zeros(cs)

    if add_entropy:
        def get_reg_coeff():
            current = start_coef
            while True:
                yield current
                if current > 0:
                    current -= decrease
        reg_coefs = get_reg_coeff()
    else:
        reg_coefs = None

    def objective(sample, reg_coef=None):
        # sample has shape (sample_size, n)
        barycenters = torch.softmax(sample, dim=-1).T.contiguous()  # shape (n, sample_size)
        if device == 'cpu':
            barycenters = barycenters.numpy()
        if barycenters.shape[-1] == 1:  # evaluate objective for single point
            barycenters = barycenters.flatten()
            single_sample = True
        else:  # evaluate objective for a sample of points
            single_sample = False

        if device == 'cpu':
            wasser_dist = [ot.emd2(c, barycenters, cost_mat) for c in cs]
        else:
            wasser_dist = [ot.sinkhorn2(c, barycenters, cost_mat, reg=1e-2) for c in cs]
        entropy = reg_coef * scipy.stats.entropy(barycenters) if add_entropy and reg_coef else 0.  # shape (sample_size,)

        if single_sample:
            objective_val = sum(wasser_dist)
        else:  # wasser_dist is list of (sample_size,) tensors
            objective_val = torch.tensor(wasser_dist, dtype=dtype) if device == 'cpu'\
                else torch.vstack(wasser_dist)
            objective_val = objective_val.sum(dim=0)
            if temperature:
                objective_val *= temperature
        return -objective_val + entropy  # minus because algorithm maximizes objective

    if noise_level is not None:
        r_prior, z_prior = get_init_barycenter(r_opt, noise_level, seed=0)  # initial approx. barycenter
    else:
        r_prior = torch.ones_like(r_opt)
        r_prior /= r_prior.sum()
        z_prior = safe_log(r_prior)

    # the following function is needed to store info about convergence of the algorithm
    def get_info(z):
        """
        Get information about how far the current point from optimum.

        :param z: preimage (in R^n) of approx. barycenter at the current step
        :return: list containing current approximation of barycenter, objective value
            and distance to the optimum
        """
        r = torch.softmax(z, dim=-1)
        acc_r = norm_sq(r - r_opt)  # distance between current approximation and true barycenter
        objective_val = -objective(z.unsqueeze(0))
        return r, objective_val, acc_r

    prior_cov = prior_var * torch.eye(img_size**2, dtype=dtype, device=device)
    get_sample = get_sampler(sample_size)

    def recalculate_cov(old_cov, sample, step, weights):
        factor = var_decay ** step if decay == 'exp' else var_decay / (step + var_decay)
        matrix = torch.cov(sample.T, aweights=weights) if empir_cov else prior_cov
        return factor * matrix

    trajectory = algorithm(z_prior, prior_cov, get_sample, n_steps, objective,
                           recalculate_cov, seed=0, get_info=get_info, track_time=track_time, hyperparam=reg_coefs)

    if plot:
        # Names and indices of elements of the list returned by 'get_info'
        info_names = [{'Objective': 1}, {'Accuracy of r': 2}]
        n_cols = 6
        img_name = f"samples_{sample_size}_var_{prior_var}_decay_{var_decay}"
        if empir_cov:
            img_name = 'cov_' + img_name
        if device == 'cuda':
            img_name = 'GPU_' + img_name

        r_opt = r_opt.numpy() if device == 'cpu' else replace_zeros(r_opt)
        opt_val = get_optimal_value(device, cost_mat, cs, r_opt)  # needed for displaying optimal value on plot
        plot_trajectory(trajectory, n_cols, img_size, img_name, info_names, opt_val=opt_val)

    else:
        return sum([info[2] for info in trajectory[-3:]]) / 3, trajectory


def pot_grid_search():
    img_size = 5  # width = height of an image
    column_interval = 1
    n_steps = 10  # number of iterations

    n_cols = 6
    info_names = [{'Objective': 1}, {'Accuracy of r': 2}]

    # Hyperparameters
    device = 'cpu'
    sample_sizes = [8 ** 3]  # [8 ** i for i in range(3, 6)]
    # sample_size = 1000
    prior_vars = [2. ** i for i in range(3, 5)]
    # prior_var = 7.  # initial variance of prior
    var_decays = [0.65, 0.8, 0.95]
    # var_decay = 0.85  # decay rate for variance

    for sample_size in sample_sizes:
        print("SAMPLE SIZE:", sample_size)
        best_acc = 10. ** 6
        best_hyperpar = None
        best_traj = None
        for prior_var in prior_vars:
            for var_decay in var_decays:
                start = time.time()
                acc_r, traj = experiment_pot(img_size, column_interval, n_steps, device, sample_size, prior_var, var_decay)
                print(f"Experiment took {time.time() - start:.2f} s")
                if acc_r < best_acc:
                    best_acc = acc_r
                    best_hyperpar = (prior_var, var_decay)
                    best_traj = traj
                else:
                    del traj
                gc.collect()
        print(f"Best prior_var: {best_hyperpar[0]}, best var_decay: {best_hyperpar[1]}")
        img_name = f"5_samples_{sample_size}_var_{best_hyperpar[0]}_decay_{best_hyperpar[1]}"
        plot_trajectory(best_traj, n_cols, img_size, img_name, info_names)

    # prior_stds = [2.5]  # test various standard deviations for sampling
    # std_decays = [0.9]  # test various decay rates of standard deviation for sampling
    # for std_decay in std_decays:
    #     for prior_std in prior_stds:
    #          experiment_pot(std_decay, sample_size, n_steps, float(prior_std), noise_level, device, add_entropy=False)


def pot_gpu():
    img_size = 5  # width = height of an image
    column_interval = 1
    n_steps = 50  # number of iterations

    # Hyperparameters
    device = 'cpu'
    sample_size = 8 ** 4
    prior_var = 8.  # initial variance of prior
    var_decay = 20.

    experiment_pot(img_size, column_interval, n_steps, device, sample_size, prior_var, var_decay, decay='lin', plot=True, track_time=True)


def main():
    # pot_grid_search()
    pot_gpu()


if __name__ == "__main__":
    main()
