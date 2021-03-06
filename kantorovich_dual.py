import torch
import time
import gc
# from itertools import product
from tqdm.contrib.itertools import product
from tqdm import tqdm

import numpy as np
import ot
from scipy.optimize import minimize

from algorithm import algorithm
from experiment_barycenter import get_data_and_solution
from utils import get_cost_mat, plot_trajectory, norm_sq, get_sampler, show_barycenters, get_empir_cov, scale_cov, replace_zeros

from Wass import algorithm as vaios_alg
from Wass import Objective


def get_c_concave(phi, cost_mat, double=True):
    # 'phi' has size (M, m*n), where M is sample size or 1
    M = phi.shape[0]
    n = cost_mat.shape[0]
    m = phi.shape[1] // n
    phi_c, _ = (cost_mat - phi.reshape(M, m, n, 1)).min(dim=2)  # (M, m, n)
    if double:
        phi_cc, _ = (cost_mat - phi_c.reshape(M, m, 1, n)).min(dim=3)  # (M, m, n)
        return phi_c, phi_cc
    return phi_c, phi.reshape(M, m, n)


def objective_function(sample, cost_mat, cs, kappa, return_what=None,
                       double_conjugate=True, inverse_order=False, kappa2=None):
    # 'sample' has size (M, m*n), where M is sample size or 1
    if return_what is None:
        return_what = ['objective']
    result = []

    phi_c, phi_cc = get_c_concave(sample, cost_mat, double=double_conjugate)  # (M, m, n)
    if inverse_order:
        phi_c, phi_cc = phi_cc, phi_c

    phi_bar = phi_cc.sum(dim=1)  # (M, n)

    if 'objective' in return_what:
        logsumexp = -kappa * torch.logsumexp(-phi_bar / kappa, dim=-1)  # (M,)
        inner_prod = (phi_c * cs).sum(dim=(1, 2))  # (M,)
        obj_val = logsumexp + inner_prod
        result += [obj_val]

    if 'barycenter' in return_what:
        bary = torch.softmax(-phi_bar / (kappa if kappa2 is None else kappa2), dim=-1)
        result += [bary]

    return result


def run_experiment(img_size, device, prior_var, var_decay, n_steps, sample_size, kappa, decay='exp', plot=True,
                   empir_cov=False, temperature=1., track_time=False, z_prior=None, double_conjugate=True,
                   inverse_order=False):
    n = img_size ** 2  # dimensionality of barycenter
    dtype = torch.float64

    cost_mat = get_cost_mat(img_size, device, dtype=dtype)
    cs, r_opt = get_data_and_solution(device, dtype=dtype, size=img_size, column_interval=(1 if img_size == 3 else 2))
    m = cs.shape[0]

    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa,double_conjugate=double_conjugate,
                                                  inverse_order=inverse_order)[0]

    def get_info(z):
        # 'z' has size (m * n,)
        obj_val, bary = objective_function(z.unsqueeze(0), cost_mat, cs, kappa, return_what=['objective', 'barycenter'],
                                           double_conjugate=double_conjugate, inverse_order=inverse_order)
        acc_r = norm_sq(bary - r_opt)  # distance between current approximation and true barycenter
        return bary, -obj_val, acc_r

    if z_prior is None:
        z_prior = -torch.ones(m * n, device=device, dtype=dtype)

    prior_cov = prior_var * torch.eye(m * n, dtype=dtype, device=device)
    get_sample = get_sampler(sample_size)

    def recalculate_cov(old_cov, sample, step, weights):
        return get_empir_cov(sample, step, weights, decay, var_decay)\
            if empir_cov else scale_cov(step, decay, var_decay, prior_cov)

    trajectory = algorithm(z_prior, prior_cov, get_sample, n_steps, objective,
                           recalculate_cov, seed=0, get_info=get_info, temperature=temperature, track_time=track_time)

    if plot:
        # Names and indices of elements of the list returned by 'get_info'
        info_names = [{'Objective': 1}, {'Accuracy of r': 2}]
        n_cols = 6
        img_name = f"sz_{img_size}_dual2_samp_{sample_size}_var_{prior_var:.2e}_dec_{var_decay}_kap_{kappa:.2e}_temp_{temperature}"
        img_name += '_empir' if empir_cov else ''

        plot_trajectory(trajectory, n_cols, img_size, img_name, info_names, use_softmax=False, scaling='partial')
        # res = torch.stack([x[0] for x in trajectory[100:]])
        # images = [trajectory[0][0], res.mean(dim=0)]
        # show_barycenters(images, img_size, img_name, use_softmax=False, scaling='partial')
    else:
        n_average = 5
        wasser_dist = 0.
        for bary in trajectory[-n_average:]:
            for c in cs:
                wasser_dist += ot.emd2(bary[0][0], c, cost_mat)
        return wasser_dist / n_average


def best_params():
    img_size = 5
    device = 'cpu'
    n_steps = 300

    # Hyperparameters
    sample_size = 5000
    prior_var = 0.1
    var_decay = 1.
    kappa = 1. / 30
    temp = 80
    # z_prior = torch.load('phi.pt')
    # start = time.time()
    run_experiment(img_size, device, prior_var, var_decay, n_steps, sample_size, kappa, temperature=temp,
                   track_time=True, double_conjugate=False, inverse_order=True)  # , plot=False


def grid_search():
    img_size = 5
    device = 'cpu'
    n_steps = 200
    z_prior = torch.load('phi.pt')

    # Hyperparameters
    sample_size = 5000
    var_decay = 1.
    prior_vars = [10. ** (-3 + i / 2) for i in range(-1, 5)]  # 11
    temps = [0.5 * 10. ** (i / 2) for i in range(1, 7)]  # 7
    kappas = [10. ** (-2 + i / 2) for i in range(3)]  # 5

    grid = product(prior_vars, temps, kappas)
    results = []
    for var, t, kap in grid:  # tqdm(grid):
        wasser_dist = run_experiment(img_size, device, var, var_decay, n_steps, sample_size, kap, temperature=t,
                                     double_conjugate=False, inverse_order=False, plot=False, z_prior=z_prior)
        results.append((var, t, kap, wasser_dist))

    top = 10
    best = sorted(results, key=lambda x: x[-1])[:top]
    for res in best:
        var, t, kap, wass = res
        print(f"{wass:.2f} | var={var:.2e}, t={t:.2f}, kap={var:.3f}")


def test_c_concave():
    img_sz = 3
    cost_mat = get_cost_mat(img_sz, 'cpu')
    n = img_sz ** 2

    def vaios_c_concave(cost, potsample):
        lam = potsample
        n_ = len(potsample[0])
        M = len(lam)

        lamext = lam.reshape(M, n_, 1).expand(M, n_, n_).transpose(2, 1)
        lamstar = (cost - lamext).amin(dim=2)

        lamstarext = lamstar.reshape(M, n_, 1).expand(M, n_, n_).transpose(2, 1)
        lamstarstar = (cost - lamstarext).amin(dim=2)
        return lamstar, lamstarstar

    sample_size = 10
    mean = torch.zeros(n)
    cov = 100 * torch.eye(n)
    get_sample = get_sampler(sample_size)

    for i in range(10):
        sample = get_sample(mean, cov, i)
        phi_c, phi_cc = get_c_concave(sample, cost_mat)
        lamstar, lamstarstar = vaios_c_concave(cost_mat, sample)
        err1 = norm_sq(phi_c.squeeze(1) - lamstar)
        err2 = norm_sq(phi_cc.squeeze(1) - lamstarstar)
        print(f"Error: {max(err1, err2):.2e}")


def bary_potentials():
    device = 'cpu'
    img_sz = 5
    dtype = torch.float64
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)
    n = img_sz ** 2
    cs, r_opt = get_data_and_solution(device, size=img_sz, dtype=dtype, column_interval=(1 if img_sz == 3 else 2))

    mu1 = cs[0]
    mu2 = cs[1]
    nu = r_opt

    pot_mean = torch.ones(n, dtype=dtype)
    pot_cov = torch.eye(n, dtype=dtype)

    # (potmean, potcov, potsamplesize, pot_n_steps, first, second, cost, objective, temperature, device)
    vaios_dist1, phi1 = vaios_alg(pot_mean, pot_cov, 2000, 2, mu1, nu, cost_mat, Objective, 100, device)
    vaios_dist2, phi2 = vaios_alg(pot_mean, pot_cov, 2000, 2, mu2, nu, cost_mat, Objective, 100, device)
    phi = torch.cat([phi1, phi2])
    torch.save(phi, 'bary_potentials.pt')

    # check distance by Vaios
    true_dist1 = ot.emd2(mu1, nu, cost_mat)
    true_dist2 = ot.emd2(mu2, nu, cost_mat)
    err1 = norm_sq(vaios_dist1 - true_dist1)
    err2 = norm_sq(vaios_dist2 - true_dist2)
    print(f"Error of calculated distance: {max(err1, err2):.2e}")

    bary = objective_function(phi.unsqueeze(0), cost_mat, cs, 1., return_what=['barycenter'])[0]
    title = 'Barycenter reconstructed from potentials'
    show_barycenters([bary], img_sz, 'bary_potentials', use_softmax=False, iterations=[title])


def external_maximizer():
    device = 'cpu'
    img_sz = 5
    dtype = torch.float64
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)
    n = img_sz ** 2
    cs, r_opt = get_data_and_solution(device, dtype=dtype, size=img_sz, column_interval=(1 if img_sz == 3 else 2))
    m = cs.shape[0]
    kappa = 0.1

    def objective(x):
        z = torch.from_numpy(x)
        obj_val = objective_function(z.unsqueeze(0), cost_mat, cs, kappa)[0]
        return -obj_val.item()

    x0 = torch.ones(m * n, device=device, dtype=dtype)
    res = minimize(objective, x0.numpy())
    phi = torch.from_numpy(res.x)
    torch.save(phi, 'bary_potentials.pt')

    init_bary = objective_function(x0.unsqueeze(0), cost_mat, cs, 1., return_what=['barycenter'])[0]
    bary = objective_function(phi.unsqueeze(0), cost_mat, cs, 1., return_what=['barycenter'])[0]
    titles = ['Data point 1', 'Data point 2', 'True barycenter', 'Initial point', 'Result of scipy.optimize.minimize']
    images = [cs[0], cs[1], r_opt, init_bary, bary]
    # show_barycenters(images, img_sz, 'scipy_no_scaling', use_softmax=False, iterations=titles, scaling='none')
    show_barycenters(images, img_sz, 'scipy_partial_scaling', use_softmax=False, iterations=titles, scaling='partial')
    # show_barycenters(images, img_sz, 'scipy_full_scaling', use_softmax=False, iterations=titles, scaling='full')


if __name__ == "__main__":
    # best_params()
    # grid_search()
    # test_c_concave()
    # bary_potentials()
    external_maximizer()
