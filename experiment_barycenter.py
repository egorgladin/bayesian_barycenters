"""
Experiment: find barycenter for a simple 3x3 setup.
"""
import numpy as np
import torch
import ot
import scipy.stats

from algorithm import algorithm
from utils import safe_log, plot_trajectory, norm_sq


IMG_SIZE = 3  # image size is 3x3
n = IMG_SIZE**2  # barycenter dimensionality
FLOAT64 = torch.float64  # avoid errors when converting from torch to numpy


def get_cost_matrix(im_sz, device, dtype=torch.float32):
    """
    Compute ground cost matrix for images.

    :param im_sz: positive int
    :param device: 'cpu' or 'cuda'
    :return: (im_sz^2, im_sz^2) tensor
    """
    C = torch.zeros(im_sz**2, im_sz**2, device=device, dtype=dtype)
    for i in range(im_sz):
        for j in range(im_sz):
            I = im_sz*i + j
            for k in range(im_sz):
                for l in range(im_sz):
                    J = im_sz*k + l
                    C[I, J] = (i - k)**2 + (j - l)**2
    return C / (im_sz - 1)**2


def marginals_residuals(r, Xs, cs):
    """
    Compute squared norms of residuals for marginal distributions.

    :param r: (n,) tensor (barycenter)
    :param Xs: (m, n, n) tensor (transportation plans)
    :param cs: (m, n) tensor (distributions, data points)
    :return: list of 2 scalar tensors
    """
    margin_r = Xs.sum(dim=-1) - r  # (m, n) tensor
    margin_c = Xs.sum(dim=1) - cs  # (m, n) tensor
    return norm_sq(margin_r), norm_sq(margin_c)


def barycenter_objective(r, Xs, cost_mat, kappas, cs):
    """
    Compute objective value for barycenter problem.

    :param r: (n,) tensor (barycenter)
    :param Xs: (m, n, n) tensor (transportation plans)
    :param cost_mat: (n, n) tensor
    :param kappas: list of 2 scalar tensors (kappa_r and kappa_c)
    :param cs: (m, n) tensor (distributions, data points)
    :return: scalar tensor
    """
    transport_cost = torch.tensordot(Xs, cost_mat.unsqueeze(0), dims=3)  # scalar tensor
    penalty_r, penalty_c = marginals_residuals(r, Xs, cs)  # 2 scalar tensors
    kappa_r, kappa_c = kappas
    return transport_cost + kappa_r * penalty_r + kappa_c * penalty_c


def map_to_simplex(theta, n_):
    """
    Map a vector of parameters from R^d to points on unit simplexes.

    :param theta: (d,) tensor with d = n + m*n^2
    :param n_: dimension of barycenter
    :return: list of (n,) tensor (barycenter) and (m, n, n) tensor (plans)
    """
    z = theta[:n_]
    Ys = theta[n_:].reshape(-1, n_ ** 2)
    r = torch.softmax(z, dim=-1)
    Xs = torch.softmax(Ys, dim=-1)
    return r, Xs.reshape(-1, n_, n_)


def get_data_and_solution(device, dtype=torch.float32):
    """
    Get two simple 3x3 images and their barycenter.

    :param device: 'cuda' or 'cpu'
    :return: list of (2, 9) tensor (data points) and (9,) tensor (barycenter)
    """
    ones = torch.ones((3, 1), device=device, dtype=dtype) / 3
    c1 = ones * torch.tensor([1., 0, 0], device=device, dtype=dtype)
    c2 = ones * torch.tensor([0., 0, 1], device=device, dtype=dtype)
    barycenter = ones * torch.tensor([0., 1, 0], device=device, dtype=dtype)
    cs = torch.stack([c1.flatten(), c2.flatten()])
    return cs, barycenter.flatten()


def get_optimal_plans(device):
    """
    Get optimal plans for simple 3x3 images.

    :param device: 'cuda' or 'cpu'
    :return: (2, 9, 9) tensor
    """
    im_sz = 3
    X1 = torch.zeros(im_sz**2, im_sz**2, device=device)
    X2 = torch.zeros_like(X1)
    for i, idx_in_r in enumerate([1, 4, 7]):
        X1[idx_in_r, i*3] = 1. / 3.
        X2[idx_in_r, i*3 + 2] = 1. / 3.
    return torch.stack([X1, X2])


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


def get_init_plans(r_prior, cs):
    """Define initial transport plans that give correct marginals."""
    X1 = r_prior.unsqueeze(1) @ cs[0].unsqueeze(0)
    X2 = r_prior.unsqueeze(1) @ cs[1].unsqueeze(0)
    Xs = torch.stack([X1, X2])

    # map to the whole Euclidean space
    Ys = safe_log(Xs)
    return Xs, Ys


def get_optimal_value(device, cost_mat, cs, r_opt):
    wasser_dist = [ot.emd2(c, r_opt, cost_mat) for c in cs]
    return sum(wasser_dist)


def alg_full_sampling(kappa, std_decay, sample_size, n_steps, prior_std, noise_level, device):
    cost_mat = get_cost_matrix(IMG_SIZE, device)
    cs, r_opt = get_data_and_solution(device)
    Xs_opt = get_optimal_plans(device)
    kappas = [kappa, kappa]  # equal coefficients for penalizing both marginals

    def objective(sample):
        sample_size = sample.shape[0]
        values = torch.empty(sample_size, device=device)
        for i in range(sample_size):
            r, Xs = map_to_simplex(sample[i], n)
            values[i] = barycenter_objective(r, Xs, cost_mat, kappas, cs)
        return -values  # minus because the algorithm maximizes the objective

    # Define initial approx. barycenter and transport plans as well as their counterparts in R^n
    r_prior, z_prior = get_init_barycenter(r_opt, noise_level, seed=0)
    Xs, Ys = get_init_plans(r_prior, cs)
    prior_mean = torch.cat((z_prior, Ys.flatten()))  # concatenate variables in one tensor

    # the following function is needed to store info about convergence of the algorithm
    def get_info(theta):
        """
        Get information about how far the current point from optimum.

        :param theta: vector of variables at the current step
        :return: list containing current approximation of barycenter and
            information about how far the current point from optimum
        """
        r, Xs = map_to_simplex(theta, n)
        residue_r, residue_c = marginals_residuals(r, Xs, cs)
        acc_X1 = norm_sq(Xs[0] - Xs_opt[0])  # distance between current and optimal plans
        acc_X2 = norm_sq(Xs[1] - Xs_opt[1])
        acc_r = norm_sq(r - r_opt)  # distance between current approximation and true barycenter
        objective_val = -objective(theta.unsqueeze(0)).item()
        transport_cost = torch.tensordot(Xs, cost_mat.unsqueeze(0), dims=3).item()
        return r, residue_r, residue_c, objective_val, transport_cost, acc_X1, acc_X2, acc_r

    # Names and indices of elements of the list returned by 'get_info'
    info_names = [{'Marginal error for r': 1, 'Marginal error for c': 2, 'Objective': 3, 'Transport cost': 4},
                  {'Accuracy of X1': 5, 'Accuracy of X2': 6}, {'Accuracy of r': 7}]

    trajectory = algorithm(prior_mean, prior_std, n_steps, sample_size, objective,
                           std_decay=std_decay, seed=0, get_info=get_info, track_time=True)
    n_cols = 6
    img_name = f"kappa_{kappa}_sample_size_{sample_size}_prior_std_{prior_std}"
    plot_trajectory(trajectory, n_cols, IMG_SIZE, img_name, info_names)


def alg_ot_lib(std_decay, sample_size, n_steps, prior_std, noise_level, device, add_entropy=False, start_coef=20., decrease=0.25):
    cost_mat = get_cost_matrix(IMG_SIZE, device, dtype=FLOAT64).numpy()
    cs, r_opt = get_data_and_solution(device, dtype=FLOAT64)
    cs = cs.numpy()

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
        barycenters = torch.softmax(sample, dim=-1).numpy()
        barycenters = np.ascontiguousarray(barycenters.T)  # shape (n, sample_size)
        if barycenters.shape[-1] == 1:  # evaluate objective for single point
            barycenters = barycenters.flatten()
            single_sample = True
        else:  # evaluate objective for a sample of points
            single_sample = False

        wasser_dist = [ot.emd2(c, barycenters, cost_mat) for c in cs]
        entropy = reg_coef * scipy.stats.entropy(barycenters) if add_entropy and reg_coef else 0.  # shape (sample_size,)

        if single_sample:
            objective_val = sum(wasser_dist)
        else:  # wasser_dist is list of (sample_size,) tensors
            objective_val = torch.tensor(wasser_dist, device=device, dtype=FLOAT64).sum(dim=0)
        return -objective_val + entropy  # minus because algorithm maximizes objective

    r_prior, z_prior = get_init_barycenter(r_opt, noise_level, seed=0)  # initial approx. barycenter

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

    # Names and indices of elements of the list returned by 'get_info'
    info_names = [{'Objective': 1}, {'Accuracy of r': 2}]

    trajectory = algorithm(z_prior, prior_std, n_steps, sample_size, objective,
                           std_decay=std_decay, seed=0, get_info=get_info, track_time=True, hyperparam=reg_coefs)
    n_cols = 6
    img_name = f"OT_samples_{sample_size}_std_{prior_std}_decay_{std_decay}"
    opt_val = get_optimal_value(device, cost_mat, cs, r_opt.numpy())  # needed for displaying optimal value on plot
    plot_trajectory(trajectory, n_cols, IMG_SIZE, img_name, info_names, opt_val=opt_val)


if __name__ == "__main__":
    USE_OT_LIBRARY = True

    device = 'cpu'
    noise_level = 0.4  # initial point is defined as the true barycenter + noise
    prior_stds = [2.5]  # test various standard deviations for sampling
    std_decays = [0.85]  # test various decay rates of standard deviation for sampling
    n_steps = 20  # number of iterations
    sample_size = 10000
    for std_decay in std_decays:
        for prior_std in prior_stds:
            if USE_OT_LIBRARY:
                alg_ot_lib(std_decay, sample_size, n_steps, float(prior_std), noise_level, device, add_entropy=False)
            else:
                for kappa in [1., 3.]:  # test various coefficients for penalizing marginal distributions
                    alg_full_sampling(kappa, std_decay, sample_size, n_steps, prior_std, noise_level, device)
