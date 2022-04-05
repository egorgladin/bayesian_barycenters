import time

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import ot
import pickle

from algorithm import algorithm
from utils import load_data, safe_log, plot_trajectory, replace_zeros, get_sampler, show_barycenter,\
    show_barycenters, get_sampler
from experiment_barycenter import get_cost_matrix


def experiment(n_steps, sample_size, prior_var, var_decay, noise_level=None, decay='exp', empir_cov=False,
               temperature=None, plot=False, track_time=False):
    dtype = torch.float32
    device = 'cuda'

    n_data_points = 9
    src_digit = 5
    target_digit = 3
    r_prior, cs = load_data(n_data_points, src_digit, target_digit, device, noise=noise_level)
    r_prior = torch.ones_like(r_prior)
    r_prior /= r_prior.sum()
    cs = replace_zeros(cs)

    img_size = 8
    cost_mat = get_cost_matrix(img_size, device, dtype=dtype)

    def objective(sample):
        # sample has shape (sample_size, n)
        barycenters = torch.softmax(sample, dim=-1).T.contiguous()  # shape (n, sample_size)
        wasser_dist = [ot.sinkhorn2(c, barycenters, cost_mat, reg=2e-2, numItermax=2000) for c in cs]

        # wasser_dist is list of (sample_size,) tensors
        objective_val = torch.vstack(wasser_dist).sum(dim=0)
        if temperature:
            objective_val *= temperature
        return -objective_val  # minus because algorithm maximizes objective

    z_prior = safe_log(r_prior)
    prior_cov = prior_var * torch.eye(img_size**2, dtype=dtype, device=device)
    get_sample = get_sampler(sample_size)

    def recalculate_cov(old_cov, sample, step, weights):
        factor = var_decay ** step if decay == 'exp' else var_decay / (step + var_decay)
        matrix = torch.cov(sample.T, aweights=weights) if empir_cov else prior_cov
        return factor * matrix

    trajectory = algorithm(z_prior, prior_cov, get_sample, n_steps, objective,
                           recalculate_cov, seed=0, track_time=track_time)

    if plot:
        # Names and indices of elements of the list returned by 'get_info'
        # info_names = [{'Objective': 1}]
        n_cols = 6
        img_name = f"IMG_samples_{sample_size}_var_{prior_var}_decay_{var_decay}"
        if empir_cov:
            img_name = 'cov_' + img_name

        plot_trajectory(trajectory, n_cols, img_size, img_name, None)

    else:
        return sum([info[2] for info in trajectory[-3:]]) / 3, trajectory


def baseline():
    dtype = torch.float64
    device = 'cuda'
    img_sz = 8
    n_data_points = 10
    src_digit = 0
    target_digit = 0

    calc_bary = True
    if calc_bary:

        # 1. Take a few images of same digit
        _, cs = load_data(n_data_points, src_digit, target_digit, device, dtype=dtype)

        # 2. Calculate barycenter of those images
        # cs = replace_zeros(cs).T.contiguous()
        cost_mat = get_cost_matrix(img_sz, device, dtype=dtype)
        reg = 0.001
        r = ot.barycenter(replace_zeros(cs.clone()).T.contiguous(), cost_mat, reg, numItermax=20000)  #, verbose=True
        torch.save(r, 'digit_bary.pt')

        # 3. Sample from Gaussian with mean = barycenter
        sample_size = 1000000  # 1 mil
        get_sample = get_sampler(sample_size)
        cov = 0.001 * torch.eye(len(r), dtype=dtype, device=device)
        sample = get_sample(r, cov, 0)

        # 4. Calculate empirical mean and covariance
        empir_mean = sample.mean(dim=0)
        empir_cov = torch.cov(sample.T)
        del sample
        torch.save(empir_mean, 'digit_empir_mean.pt')
        torch.save(empir_cov, 'digit_empir_cov.pt')

    else:
        empir_mean = torch.load('digit_empir_mean.pt')
        empir_cov = torch.load('digit_empir_cov.pt')
        r = torch.load('digit_bary.pt')
        _, cs = load_data(n_data_points, src_digit, target_digit, device, dtype=dtype)

    # 5. Sample using empirical mean and covariance
    get_sample = get_sampler(8)
    sample = get_sample(empir_mean, empir_cov, 1)

    images = [cs[i] for i in range(10)] + [r, empir_mean] + [sample[i] for i in range(8)]  # [::2]
    titles = [f'Data point #{i}' for i in range(1, 11)] + ['Barycenter', 'Empirical mean'] + [f'Sample #{i}' for i in range(1, 9)]
    show_barycenters(images, img_sz, 'digits4', use_softmax=False, iterations=titles, scaling='partial')


def main():
    n_steps = 300
    sample_size = 30000
    prior_var = 8.
    var_decay = 20.
    experiment(n_steps, sample_size, prior_var, var_decay, noise_level=0.1, decay='lin',
               plot=True, track_time=True)


if __name__ == "__main__":
    # main()
    baseline()
