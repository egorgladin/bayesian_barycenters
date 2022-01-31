import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.dirichlet import Dirichlet
import ot
from experiment_barycenter import get_cost_matrix

import time
import csv
import pickle


def gen_data(distr, size, seed1, seed2):
    torch.manual_seed(seed1)
    a = distr.sample()
    torch.manual_seed(seed2)
    b = distr.sample((size,))
    return a, b.T.contiguous()


def change_dtype(a, b, new_dtype):
    a = a.type(new_dtype)
    a /= a.sum()
    b = b.type(new_dtype)
    b /= b.sum(axis=0, keepdims=True)
    return a, b


def sinkhorn_comparison():
    n_experim = 2

    im_sz = 5
    n_samples = 10
    cost_mat = get_cost_matrix(im_sz, device='cuda', dtype=torch.float32)

    methods = ['sinkhorn', 'sinkhorn_stabilized', 'sinkhorn_log']
    reg_coeffs = [1e-1, 1e-2, 1e-3]  #
    results = {method: {reg_coeff: {'time': 0., 'relative_eror': 0.} for reg_coeff in reg_coeffs}
               for method in methods}
    exact_emds = []

    alpha = torch.ones(im_sz**2, device='cuda', dtype=torch.float32)
    distr = Dirichlet(alpha)

    # exact emd
    start = time.time()
    for i in range(n_experim):
        a, b = gen_data(distr, n_samples, i, n_experim + i)
        a, b = change_dtype(a, b, torch.float64)
        emds = ot.emd2(a, b, cost_mat)
        exact_emds.append(sum(emds))

    time_emd = (time.time() - start) / n_experim
    print(f'Time emd: {time_emd:.3f}s')  # Time emd: 19.778s

    # sinkhorn
    for reg_coeff in reg_coeffs:
        for alg in methods:
            start = time.time()
            for i in range(n_experim):
                a, b = gen_data(distr, n_samples, i, n_experim + i)
                emds = ot.sinkhorn2(a, b, cost_mat, method=alg, reg=reg_coeff)
                error = (sum(emds) - exact_emds[i]) / exact_emds[i]
                results[alg][reg_coeff]['relative_eror'] += abs(error.item()) / n_experim

            results[alg][reg_coeff]['time'] = (time.time() - start) / n_experim
            print(f"Time (reg {reg_coeff}, alg {alg}): {results[alg][reg_coeff]['time']:.3f}s")

    for metrics in ['time', 'relative_eror']:
        with open(f'solvers_comparison_{im_sz}x{im_sz}_{metrics}.csv', 'w', newline='') as csvfile:
            fieldnames = ['reg_coeff'] + methods
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for reg_coeff in reg_coeffs:
                row = {alg: results[alg][reg_coeff][metrics] for alg in methods}
                row['reg_coeff'] = reg_coeff
                writer.writerow(row)


def imsz2nsamples(x):
    return np.square(x)


def nsamples2imsz(x):
    return np.sqrt(x)


def plot_increasing_dim(metric, im_szs, results):
    fig, ax = plt.subplots(constrained_layout=True)
    for name, vals in results.items():
        if name == 'sinkhorn_log, reg_coeff=0.01':
            ax.plot(im_szs, vals, label=name, linestyle='--')
        else:
            ax.plot(im_szs, vals, label=name, linewidth=3, alpha=0.7)
    ax.set_xlabel('Image size')
    ax.set_ylabel(metric)
    plt.legend()
    # ax.set_title('Sine wave')

    secax = ax.secondary_xaxis('top', functions=(imsz2nsamples, nsamples2imsz))
    secax.set_xlabel('Number of samples')
    plt.savefig(f"plots/wasser_dist_{metric}.png", bbox_inches='tight')
    plt.close()


def sinkhorn_scaling():
    n_experim = 10
    methods = [('sinkhorn', 1e-2), ('sinkhorn_log', 1e-2), ('sinkhorn_log', 5e-3)]
    times = {}
    errors = {}

    im_szs = [5, 10, 15, 20]
    for j, im_sz in enumerate(im_szs):
        print(f'IMAGE SIZE {im_sz}')
        n_samples = im_sz ** 2
        cost_mat = get_cost_matrix(im_sz, device='cuda', dtype=torch.float32)
        exact_emds = []

        alpha = torch.ones(im_sz**2, device='cuda', dtype=torch.float32)
        distr = Dirichlet(alpha)

        # exact emd
        name = 'exact emd'
        if name not in times:
            times[name] = [0.] * len(im_szs)
        start = time.time()
        for i in range(n_experim):
            a, b = gen_data(distr, n_samples, i, n_experim + i)
            a, b = change_dtype(a, b, torch.float64)
            emds = ot.emd2(a, b, cost_mat)
            exact_emds.append(sum(emds))

        times[name][j] = (time.time() - start) / n_experim
        print(f'Time emd: {times[name][j]:.3f}s')

        for alg, reg_coeff in methods:
            name = f'{alg}, reg_coeff={reg_coeff}'
            if name not in errors:
                errors[name] = [0.] * len(im_szs)
                times[name] = [0.] * len(im_szs)

            start = time.time()
            for i in range(n_experim):
                a, b = gen_data(distr, n_samples, i, n_experim + i)
                emds = ot.sinkhorn2(a, b, cost_mat, method=alg, reg=reg_coeff)
                error = (sum(emds) - exact_emds[i]) / exact_emds[i]
                errors[name][j] += abs(error.item()) / n_experim

            times[name][j] = (time.time() - start) / n_experim
            print(f"Time (reg {reg_coeff}, alg {alg}): {times[name][j]:.2f}s")

        with open('pickled/wasser_dist_performance.pickle', 'wb') as handle:
            pickle.dump((times, errors), handle)
    plot_increasing_dim('Time, s', im_szs, times)
    plot_increasing_dim('Relative error', im_szs, errors)


def main():
    # sinkhorn_comparison()
    # sinkhorn_scaling()

    with open('pickled/wasser_dist_performance.pickle', 'rb') as handle:
        times, errors = pickle.load(handle)
    im_szs = [5, 10, 15, 20]
    plot_increasing_dim('Relative error', im_szs, errors)


if __name__ == "__main__":
    main()
