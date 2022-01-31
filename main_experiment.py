"""
Experiment: find barycenter for a simple 3x3 setup.
"""
import numpy as np
import torch
import ot

from algorithm import algorithm
from two_step_algorithm import two_step_algorithm
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



def alg_ot_lib(std_decay, sample_size, n_steps, prior_std, noise_level, device, alg):
    cost_mat = get_cost_matrix(IMG_SIZE, device, dtype=FLOAT64).numpy()
    cs, r_opt = get_data_and_solution(device, dtype=FLOAT64)
    cs = cs.numpy()

    def objective(sample):
        barycenters = sample.clone()  #sample is already transformed onto the simplex in algorithm
        barycenters = torch.softmax(sample, dim=-1).numpy() #comment this line if algorithm returns transformed var
        barycenters = np.ascontiguousarray(barycenters.T)
        if barycenters.shape[-1] == 1:  # evaluate objective for single point
            barycenters = barycenters.flatten()
            single_sample = True
        else:  # evaluate objective for a sample of points
            single_sample = False

        wasser_dist = [ot.emd2(c, barycenters, cost_mat) for c in cs]

        if single_sample:
            objective_val = -sum(wasser_dist)
        else:
            objective_val = -torch.tensor(wasser_dist, device=device, dtype=FLOAT64).sum(dim=0)
        return objective_val  # minus because the algorithm maximizes the objective

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
    if alg == 1:
        trajectory = algorithm(z_prior, prior_std, n_steps, sample_size, objective, 
                               std_decay=std_decay, seed=0, get_info=get_info, track_time=True)
    else :
            trajectory = two_step_algorithm(z_prior, prior_std, n_steps, sample_size, objective, 
                                            std_decay=std_decay, seed=0, get_info=get_info, track_time=True)
            
    
    n_cols = 6
    img_name = f"OT_sample_size_{sample_size}_prior_std_{prior_std}_alg_{alg}"
    opt_val = get_optimal_value(device, cost_mat, cs, r_opt.numpy())  # needed for displaying optimal value on plot
    plot_trajectory(trajectory, n_cols, IMG_SIZE, img_name, info_names, opt_val=opt_val)


if __name__ == "__main__":
    USE_OT_LIBRARY = True

    device = 'cpu'
    std_decay = 0.98  # decay rate of standard deviation for sampling
    noise_level = 2.  # initial point is defined as the true barycenter + noise
    n_steps = 50  # number of iterations
    sample_size = 1000
    #for prior_std in [2.5, 3., 4.]:  # test various standard deviations for sampling
    prior_std = 1.1   
    alg = 2  # 1 for one-step algorithm, 2 for two-step algorithm
    alg_ot_lib(std_decay, sample_size, n_steps, prior_std, noise_level, device, alg)
    #if USE_OT_LIBRARY:

        
       # else:
           # for kappa in [1., 3.]:  # test various coefficients for penalizing marginal distributions
               # alg_full_sampling(kappa, std_decay, sample_size, n_steps, prior_std, noise_level, device)
