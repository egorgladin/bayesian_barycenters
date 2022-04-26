import time
from os import environ
from math import ceil
import sys

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import ot
import pickle
from tqdm import tqdm
from operator import itemgetter

from algorithm import algorithm
from utils import get_cost_mat, load_data, safe_log, plot_trajectory, replace_zeros, get_sampler, show_barycenter, \
    show_barycenters, get_sampler, norm_sq, get_digits_and_bary
from posterior_mean_and_cov import get_posterior_mean_cov

from Wass import algorithm as vaios_alg
from Wass import Objective
from get_potentials import algorithm as alg_for_poten
from kantorovich_dual import objective_function


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
    cost_mat = get_cost_mat(img_size, device, dtype=dtype)

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
    prior_cov = prior_var * torch.eye(img_size ** 2, dtype=dtype, device=device)
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


def get_poten_from_pot(cs, r, cost_mat, reverse_order, device, calc_poten_method='emd', reg=None):
    us = []
    vs = []
    emds = []
    for c in tqdm(cs):
        if reverse_order:
            mu, nu = c, r
        else:
            mu, nu = r, c
        res = ot.emd(mu, nu, cost_mat, log=True)[1] if calc_poten_method == 'emd' \
            else ot.sinkhorn(mu, nu, cost_mat, reg, numItermax=20000, log=True)[1]
        u, v = res['u'], res['v']
        us.append(u.clone())
        vs.append(v.clone())
        cur_emd = u @ mu + v @ nu
        emds.append(cur_emd.item())
        del res
        if device == 'cuda':
            torch.cuda.empty_cache()

    potentials = torch.stack(us + vs)  # (2 * n_data_points, n)
    return potentials, emds


def get_poten_from_vaios(wass_params, cs, r, cost_mat, reverse_order, device, dtype):
    n = len(r)
    n_samples, n_steps, temp, pot_var = itemgetter('n_samples', 'n_steps', 'temp', 'pot_var')(wass_params)
    pot_mean = torch.ones(n, dtype=dtype, device=device)
    pot_cov = pot_var * torch.eye(n, dtype=dtype, device=device)

    potentials = []
    vaios_distances = []
    for c in tqdm(cs):
        if reverse_order:
            mu, nu = c, r
        else:
            mu, nu = r, c
        vaios_dist, phi = vaios_alg(pot_mean, pot_cov, n_samples, n_steps, mu, nu, cost_mat, Objective, temp, device)
        potentials.append(phi.clone())
        vaios_distances.append(vaios_dist.item())
        del phi, vaios_dist
        if device == 'cuda':
            torch.cuda.empty_cache()

    potentials = torch.stack(potentials)  # (n_data_points, n)
    return potentials, vaios_distances


def get_bary_from_poten(potentials, kappa, r, get_error=True):
    bary_from_poten = torch.softmax(-potentials.sum(dim=0) / kappa, dim=0)
    if get_error:
        bary_err = torch.norm(bary_from_poten - r) / torch.norm(r)
        print(f"Relative error of barycenter recovered from potentials: {100 * bary_err:.2f}%")
    return bary_from_poten


def get_poten_from_alg(cs, cost_mat, device):
    m, n = cs.shape
    Mean = torch.ones(m * n, dtype=torch.float64, device=device)
    Covariance = torch.eye(m * n, dtype=torch.float64, device=device)
    potentials = alg_for_poten(Mean, Covariance, 10000, 5000, cs, cost_mat, device, 40)
    return potentials


def mnist_experiment(sample_size, n_batches, prior_var=1e-5, device='cuda', temperature=30., kappa=1./30):
    folder = 'digit_experiment/'
    img_sz = 28

    cs = torch.load(folder + 'Archetypes.pt', map_location=device)
    if cs.requires_grad:
        cs = cs.detach()
    dtype = cs.dtype
    print(f"dtype: {dtype}, cs shape: {cs.shape}")
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)
    potentials = torch.load(folder + 'Mean.pt', map_location=device)
    if potentials.requires_grad:
        potentials = potentials.detach()

    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa,
                                                  inverse_order=True, double_conjugate=False)[0]

    increment = ceil(sample_size / n_batches)
    # get_sample = get_sampler(increment)
    # cov = prior_var * torch.eye(potentials.numel(), dtype=dtype, device=device)
    print('Sampling with variance', prior_var)
    prior_std = torch.sqrt(torch.tensor(prior_var, device=device, dtype=dtype))

    def sample_generator():
        prior_mean = potentials.reshape(1, -1).expand(increment, -1)
        for i in range(n_batches):
            print(f"sampling batch {i}")
            torch.manual_seed(i)
            yield torch.normal(prior_mean, prior_std)

    result_mean, result_cov = get_posterior_mean_cov(sample_generator, objective, temperature,
                                                     save_covs=True, save_dir=folder, total=n_batches)
    info = f'_k{kappa}_t{temperature}_var{prior_var}_N{sample_size}.pt'
    torch.save(result_mean, folder + 'mean' + info)
    torch.save(result_cov, folder + 'cov' + info)
    print("Saved mean and cov")


def baseline(wass_params, kappa, device='cuda', calc_poten_method='sinkhorn',
             calc_poten=True, reverse_order=False, do_sampling=False, sample_size=1000000,
             prior_var=0.001, n_batches=100, calc_weights=True, temperature=10., calc_posterior=True, mnist=False):
    assert calc_poten_method in ['emd', 'vaios', 'alg', 'sinkhorn']
    assert sample_size % n_batches == 0
    dtype = torch.float64
    folder = 'digit_experiment/'
    img_sz = 8
    n_data_points = 5
    src_digit = 0
    target_digit = 0
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)

    # 1. Get a few images of the same digit and their barycenter
    if mnist:
        cs = torch.load(folder + 'Archetypes.pt')
        r = torch.load(folder + 'mnist_bary_35_im_of_5.pt')
    else:
        cs, r = get_digits_and_bary(folder + 'digits.pt', folder + 'barycenter.pt', target_digit=target_digit,
                                    n_data_points=n_data_points, dtype=dtype, device=device, cost_mat=cost_mat,
                                    verbose=True)

    n = len(r)

    # 3. Find the respective potentials
    if calc_poten:
        if calc_poten_method == 'sinkhorn':
            print("Calculating potentials with Sinkhorn")
            reg = 0.001
            potentials = get_poten_from_pot(cs, r, cost_mat, reverse_order, device,
                                            calc_poten_method='sinkhorn', reg=reg)  # (2 * n_data_points, n)
        else:
            if calc_poten_method == 'emd':
                print("Calculating potentials with ot.emd")
                potentials, distances = get_poten_from_pot(cs, r, cost_mat, reverse_order,
                                                           device)  # (2 * n_data_points, n)
            elif calc_poten_method == 'vaios':
                print("Calculating potentials with Vaios' program for Wasserstein distance")
                potentials, distances = get_poten_from_vaios(wass_params, cs, r, cost_mat, reverse_order, device, dtype)
            elif calc_poten_method == 'alg':
                print("Calculating potentials with Vaios' algorithm")
                potentials = get_poten_from_alg(cs, cost_mat, device)  # (n_data_points, n)

            with open(folder + 'approx_distances.pickle', 'wb') as handle:
                pickle.dump(distances, handle)
                print("Saved approximate distances to 'approx_distances.pickle'")

        torch.save(potentials, folder + 'potentials.pt')
        print("Saved potentials to 'potentials.pt'")
    else:
        potentials = torch.load(folder + ('MeanNew.pt' if mnist else 'potentials.pt'), map_location=device)
        print("Loaded potentials from 'potentials.pt'")
        if calc_poten_method in ['vaios', 'emd']:
            with open(folder + 'approx_distances.pickle', 'rb') as handle:
                distances = pickle.load(handle)
                print("Loaded distances from 'approx_distances.pickle'")

    # 4. Check the quality of potentials by calculating the respective primal variable
    check_poten = False
    if check_poten:
        titles = ['Barycenter (Sinkhorn)', 'Barycenter from potentials']

        if calc_poten_method in ['emd', 'sinkhorn']:
            bary_from_poten1 = get_bary_from_poten(potentials[:n_data_points], kappa, r)
            bary_from_poten2 = get_bary_from_poten(potentials[n_data_points:], kappa, r)
            barys_from_poten = [bary_from_poten1, bary_from_poten2]
            titles += ['Barycenter from potentials 2']

        if calc_poten_method in ['alg', 'vaios', 'emd']:
            if calc_poten_method in ['emd', 'vaios']:
                try:
                    handle = open(folder + 'true_distances.pickle', 'rb')
                    print("Loading true distances")
                    true_distances = pickle.load(handle)

                except FileNotFoundError:
                    print("Calculating true distances")
                    true_distances = [ot.emd2(r, c, cost_mat).item() for c in cs]
                    with open(folder + 'true_distances.pickle', 'wb') as handle:
                        pickle.dump(true_distances, handle)

                dist_errors = [abs(emd_pot - emd_vaios) for emd_pot, emd_vaios in zip(true_distances, distances)]
                dist_rel_err = sum(dist_errors) / sum(true_distances)
                print(f"Relative error of distance: {100 * dist_rel_err:.2f}%")

            if calc_poten_method in ['alg', 'vaios']:
                bary_from_poten = get_bary_from_poten(potentials, kappa, r)
                barys_from_poten = [bary_from_poten]

        show_barycenters([r] + barys_from_poten, img_sz, folder + 'bary_from_poten', use_softmax=True,
                         iterations=titles, use_default_folder=False)

    # 5. Sample potentials and calculate empirical mean and covariance
    if do_sampling:
        get_sample = get_sampler(sample_size)

        cov = prior_var * torch.eye(n_data_points * n, dtype=dtype, device=device)
        print('Sampling with variance', prior_var)

        sample = get_sample(potentials.flatten(), cov, 2)

        print('Saving the whole sample')
        torch.save(sample, folder + 'whole_sample.pt')

    else:
        print('Loading the sample')
        sample = torch.load(folder + 'whole_sample.pt', map_location=device)

    if calc_weights:
        increment = sample_size // n_batches
        objective_list = []
        print('Started calculating objective for batches')
        for i in tqdm(range(n_batches)):
            objective_list.append(
                objective_function(sample[i * increment:(i + 1) * increment], cost_mat, cs, kappa,
                                   double_conjugate=False)[0]
            )

        all_objective_vals = torch.cat(objective_list)
        print('Calculating weights with temp =', temperature)
        all_weights = torch.softmax(temperature * all_objective_vals, dim=-1)
        torch.save(all_weights, folder + 'all_weights.pt')
    else:
        print('Loading the weights')
        all_weights = torch.load(folder + 'all_weights.pt', map_location=device)

    if calc_posterior:
        empir_mean = all_weights @ sample
        empir_cov = torch.cov(sample.T, aweights=all_weights)

        del sample
        del all_weights
        if device == 'cuda':
            torch.cuda.empty_cache()

        torch.save(empir_mean, folder + 'empir_mean.pt')
        torch.save(empir_cov, folder + 'empir_cov.pt')
    else:
        empir_mean = torch.load(folder + 'empir_mean.pt', map_location=device)
        empir_cov = torch.load(folder + 'empir_cov.pt', map_location=device)

    # 6. Sample using empirical mean and covariance
    get_sample = get_sampler(5)
    sample = get_sample(empir_mean, empir_cov, 3)

    empir_mean_bary = get_bary_from_poten(empir_mean.reshape(n_data_points, n), kappa, r, get_error=False)

    images = [cs[i] for i in range(n_data_points)] + [r, empir_mean_bary]
    images += [get_bary_from_poten(sample[i].reshape(n_data_points, n), kappa, r, get_error=False) for i in
               range(sample.shape[0])]  # [::2]
    titles = [f'Data point #{i}' for i in range(1, n_data_points + 1)] + ['Barycenter', 'Empirical mean'] + [
        f'Sample #{i}' for i in range(1, sample.shape[0] + 1)]
    show_barycenters(images, img_sz, folder + 'results3', use_softmax=False, iterations=titles, scaling='partial',
                     use_default_folder=False)
    print("Saved result to 'results3.png'")


def main():
    n_steps = 300
    sample_size = 30000
    prior_var = 8.
    var_decay = 20.
    experiment(n_steps, sample_size, prior_var, var_decay, noise_level=0.1, decay='lin',
               plot=True, track_time=True)


if __name__ == "__main__":
    environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = '2570000000'
    # kappa = 1. / 30.
    # calc_poten_method = 'alg'  # potentials were calculated using Vaios' algorithm
    # calc_poten = False  # don't calculate potentials again, just load them from memory
    # do_sampling = True  # don't draw a large number of samples, just load empirical mean and cov from memory
    # baseline(None, kappa, device='cpu', calc_poten_method=calc_poten_method, calc_poten=calc_poten, reverse_order=False,
    #          do_sampling=do_sampling, sample_size=1000000, n_batches=200, prior_var=5e-5, temperature=30.)

    # sample_size = 2
    # n_batches = 2
    sample_size, n_batches = [int(sys.argv[1]), int(sys.argv[2])] if len(sys.argv) == 3 else [2, 2]
    mnist_experiment(sample_size, n_batches, device='cuda')
