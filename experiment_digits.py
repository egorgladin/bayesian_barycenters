import time

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import ot
import pickle
from tqdm import tqdm
from operator import itemgetter

from algorithm import algorithm
from utils import load_data, safe_log, plot_trajectory, replace_zeros, get_sampler, show_barycenter,\
    show_barycenters, get_sampler, norm_sq
from experiment_barycenter import get_cost_matrix

from Wass import algorithm as vaios_alg
from Wass import Objective
from get_potentials import algorithm as alg_for_poten


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


def get_poten_from_pot(cs, r, cost_mat, reverse_order, device, calc_poten_method='emd', reg=None):
    us = []
    vs = []
    emds = []
    for c in tqdm(cs):
        if reverse_order:
            mu, nu = c, r
        else:
            mu, nu = r, c
        res = ot.emd(mu, nu, cost_mat, log=True)[1] if calc_poten_method == 'emd'\
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
    Mean = torch.ones(m*n, dtype=torch.float64, device=device)
    Covariance = torch.eye(m*n, dtype=torch.float64, device=device)
    potentials = alg_for_poten(Mean, Covariance, 10000, 5000, cs, cost_mat, device, 40)
    return potentials


def baseline(wass_params, kappa, device='cuda', calc_poten_method='sinkhorn', calc_poten=True, reverse_order=False, do_sampling=False):
    assert calc_poten_method in ['emd', 'vaios', 'alg', 'sinkhorn']
    dtype = torch.float64
    folder = 'digit_experiment/'
    img_sz = 8
    n_data_points = 5
    src_digit = 0
    target_digit = 0
    cost_mat = get_cost_matrix(img_sz, device, dtype=dtype)

    # 1. Take a few images of the same digit
    try:
        cs = torch.load(folder + 'digits.pt', map_location=device)
        print("Loaded digits from 'digits.pt'")
    except FileNotFoundError:
        _, cs = load_data(n_data_points, src_digit, target_digit, device, dtype=dtype)
        torch.save(cs, folder + 'digits.pt')
        print("Obtained digits and saved to 'digits.pt'")

    # 2. Calculate barycenter of those images
    try:
        r = torch.load(folder + 'barycenter.pt', map_location=device)
        print("Loaded barycenter from 'barycenter.pt'")
    except FileNotFoundError:
        reg = 0.001
        r = ot.barycenter(replace_zeros(cs.clone()).T.contiguous(), cost_mat, reg, numItermax=20000)  #, verbose=True
        torch.save(r, folder + 'barycenter.pt')
        print("Obtained barycenter and saved to 'barycenter.pt'")

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
                potentials, distances = get_poten_from_pot(cs, r, cost_mat, reverse_order, device)  # (2 * n_data_points, n)
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
        potentials = torch.load(folder + 'potentials.pt', map_location=device)
        print("Loaded potentials from 'potentials.pt'")
        if calc_poten_method in ['vaios', 'emd']:
            with open(folder + 'approx_distances.pickle', 'rb') as handle:
                distances = pickle.load(handle)
                print("Loaded distances from 'approx_distances.pickle'")

    # 4. Check the quality of potentials by calculating the respective primal variable
    check_poten = True
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

        show_barycenters([r] + barys_from_poten, img_sz, folder + 'bary_from_poten', use_softmax=True, iterations=titles, use_default_folder=False)

    # 5. Sample potentials and calculate empirical mean and covariance
    if do_sampling:
        sample_size = 1000000  # 1 mil
        get_sample = get_sampler(sample_size)
        cov = 0.001 * torch.eye(n_data_points * n, dtype=dtype, device=device)
        sample = get_sample(potentials.flatten(), cov, 0)

        empir_mean = sample.mean(dim=0)
        empir_cov = torch.cov(sample.T)
        del sample
        if device == 'cuda':
            torch.cuda.empty_cache()

        torch.save(empir_mean, folder + 'empir_mean.pt')
        torch.save(empir_cov, folder + 'empir_cov.pt')
    else:
        empir_mean = torch.load(folder + 'empir_mean.pt', map_location=device)
        empir_cov = torch.load(folder + 'empir_cov.pt', map_location=device)

    # 6. Sample using empirical mean and covariance
    get_sample = get_sampler(5)
    sample = get_sample(empir_mean, empir_cov, 1)

    empir_mean_bary = get_bary_from_poten(empir_mean.reshape(n_data_points, n), kappa, r, get_error=False)

    images = [cs[i] for i in range(n_data_points)] + [r, empir_mean_bary]
    images += [get_bary_from_poten(sample[i].reshape(n_data_points, n), kappa, r, get_error=False) for i in
               range(sample.shape[0])]  # [::2]
    titles = [f'Data point #{i}' for i in range(1, n_data_points + 1)] + ['Barycenter', 'Empirical mean'] + [
        f'Sample #{i}' for i in range(1, sample.shape[0] + 1)]
    show_barycenters(images, img_sz, folder + 'results', use_softmax=False, iterations=titles, scaling='partial', use_default_folder=False)
    print("Saved result to 'results.png'")


def main():
    n_steps = 300
    sample_size = 30000
    prior_var = 8.
    var_decay = 20.
    experiment(n_steps, sample_size, prior_var, var_decay, noise_level=0.1, decay='lin',
               plot=True, track_time=True)


if __name__ == "__main__":
    kappa = 1. / 30.
    calc_poten_method = 'alg'  # potentials were calculated using Vaios' algorithm
    calc_poten = False  # don't calculate potentials again, just load them from memory
    do_sampling = False  # don't draw a large number of samples, just load empirical mean and cov from memory
    baseline(None, kappa, device='cpu', calc_poten_method=calc_poten_method, calc_poten=calc_poten, reverse_order=False,
             do_sampling=do_sampling)
