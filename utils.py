"""
Auxiliary functions.
"""
import torch
import numpy as np
from math import ceil
from ot import barycenter as ot_barycenter
import collections
import itertools
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter, MaxNLocator

import pickle
from sklearn.datasets import load_digits
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.datasets import MNIST
from torchvision.transforms import Resize, ToTensor, Compose


def get_cost_mat(im_sz, device, dtype=torch.float32):
    partition = torch.linspace(0, 1, im_sz)
    couples = np.array(np.meshgrid(partition, partition)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    x = torch.tensor(x, dtype=dtype, device=device)
    a = x[:, 0]
    b = x[:, 1]
    C = torch.linalg.norm(a - b, axis=1) ** 2
    return C.reshape((im_sz**2, -1))


def safe_log(arr, minus_inf=-100.):
    """Elementwise logarithm with log(0) defined by minus_inf."""
    minus_inf_tensor = minus_inf * torch.ones_like(arr)
    return torch.where(arr > 0, torch.log(arr), minus_inf_tensor)


def norm_sq(tensor):
    """Squared Euclidean norm."""
    return torch.square(tensor).sum().item()


def show_barycenter(r, fname):
    img = r.cpu().numpy().reshape(8, -1)
    plt.imshow(img, cmap='binary')
    plt.savefig(f"plots/{fname}.png", bbox_inches='tight')
    plt.close()


def show_barycenters(barycenters, img_sz, img_name, iterations=None, use_softmax=True, scaling='full', use_default_folder=True):
    """Display several barycenters across iterations."""
    n_bary = len(barycenters)
    if n_bary > 10:
        nrows, ncols = 2, ceil(n_bary / 2)
    else:
        nrows, ncols = 1, n_bary
    figsize = (ncols * 3, nrows * 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)  # , figsize=(16, 4)
    for i, ax in enumerate(axes.flat):
        if i < n_bary:
            z = barycenters[i]
            img = (torch.softmax(z, dim=-1) if use_softmax else z).cpu().numpy().reshape(img_sz, -1)
            # ax = next(axes.flat)  # axes[i] if n_bary > 1 else axes
            if np.allclose(img, img[0, 0]) or scaling == 'none':
                ax.imshow(img, cmap='binary', vmin=0, vmax=1)
            elif scaling == 'partial':
                ax.imshow(img, cmap='binary', vmin=0)
            else:
                ax.imshow(img, cmap='binary')

            if iterations is not None:
                it = iterations[i]
                title = f"Iteration {it}" if isinstance(it, int) else it
                ax.title.set_text(title)
        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])


    if use_default_folder:
        img_name = f"plots/bary_{img_name}"
    plt.savefig(img_name + ".png", bbox_inches='tight')
    plt.close()


def get_digits_and_bary(data_path, bary_path, target_digit=None, n_data_points=None,
                        dtype=torch.float32, device='cpu', cost_mat=None, verbose=False, mnist=False):
    try:
        cs = torch.load(data_path, map_location=device)
        if verbose:
            print(f"Loaded digits from {data_path}")
    except FileNotFoundError:
        not_specified = ['target_digit'] if target_digit is None else []
        not_specified += ['n_data_points'] if n_data_points is None else []
        if not_specified:
            raise ValueError(' and '.join(not_specified) + f"should be specified when file {data_path} doesn't exist")

        cs = load_mnist(n_data_points, target_digit, device, size=(28, 28), dtype=dtype)\
            if mnist else load_data(n_data_points, 5, target_digit, device, dtype=dtype)
        torch.save(cs, data_path)
        if verbose:
            print(f"Obtained digits and saved to {data_path}")

    try:
        r = torch.load(bary_path, map_location=device)
        if verbose:
            print(f"Loaded barycenter from {bary_path}")
    except FileNotFoundError:
        if cost_mat is None:
            raise ValueError(f"Cost matrix should be given when file {bary_path} doesn't exist")
        reg = 0.001
        r = ot_barycenter(replace_zeros(cs.clone()).T.contiguous(), cost_mat, reg,
                          numItermax=20000, stopThr=1e-5, verbose=verbose)
        torch.save(r, bary_path)
        if verbose:
            print(f"Obtained barycenter and saved to {bary_path}")

    return cs, r


def plot_convergence(trajectory, img_name, info_names, log_scale=False, opt_val=None):
    """Plot info stored in trajectory."""
    n_plots = len(info_names)
    figsize = (7, 4 * n_plots)
    fig, axs = plt.subplots(n_plots, 1, figsize=figsize)
    for i, names in enumerate(info_names):
        ax = axs[i]
        for name, idx in names.items():
            ax.plot([el[idx].to('cpu').numpy() if isinstance(el[idx], torch.Tensor) else el[idx]
                     for el in trajectory], label=name)
            if name == 'Objective' and opt_val is not None:
                if isinstance(opt_val, torch.Tensor):
                    opt_val = opt_val.to('cpu').numpy()
                ax.hlines(y=opt_val, xmin=0, xmax=len(trajectory)-2, label='Optimal value')
        ax.yaxis.grid()

        ax.legend()
        if (isinstance(log_scale, collections.Sequence) and log_scale[i]) or (log_scale is True):
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"plots/convergence_{img_name}.png", bbox_inches='tight')


def plot_trajectory(trajectory, n_cols, img_sz, img_name, info_names, log_scale=False, opt_val=None,
                    use_softmax=True, scaling=None):
    """Display several barycenters and plot other info stored in trajectory."""
    with open(f'pickled/trajectory_{img_name}.pickle', 'wb') as handle:
        pickle.dump(trajectory, handle)

    n_steps = len(trajectory) - 1
    # plot_every = int(n_steps / n_cols) + 1 if n_steps % n_cols else int(n_steps / n_cols)
    slope = n_steps / (n_cols - 1)
    iterations = [int(i * slope) for i in range(n_cols)]
    barycenters = [trajectory[it][0].to('cpu') for it in iterations] if info_names is not None\
        else [trajectory[it].to('cpu') for it in iterations]
    show_barycenters(barycenters, img_sz, img_name, iterations=iterations, use_softmax=use_softmax, scaling=scaling)
    if info_names is not None:
        plot_convergence(trajectory, img_name, info_names, log_scale=log_scale, opt_val=opt_val)


def compare_trajectories(file_names, plot_names, info_names, n_cols, img_sz):
    trajs = []
    for name in file_names:
        with open(f'{name}.pickle', 'rb') as handle:
            trajs.append(pickle.load(handle))

    n_steps = len(trajs[0]) - 1
    slope = n_steps / (n_cols - 1)
    iterations = [int(i * slope) for i in range(n_cols)]

    # fig, axs = plt.subplots(nrows=len(file_names), ncols=n_cols, figsize=(16, 14))
    fig, axs = plt.subplots(nrows=len(file_names), ncols=1, constrained_layout=True)

    for ax in axs:
        ax.remove()
    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    for row, subfig in enumerate(subfigs):
        barycenters = [trajs[row][it][0] for it in iterations]
        subfig.suptitle(r'$\bf{' + plot_names[row].replace(" ", "\ ") + "}$")
        axs = subfig.subplots(nrows=1, ncols=n_cols)
        for j, z in enumerate(barycenters):
            # for col, ax in enumerate(axs):
            img = torch.softmax(z, dim=-1).cpu().numpy().reshape(img_sz, -1)
            axs[j].imshow(img, cmap='binary')
            axs[j].set_title(f"step {iterations[j]}")
            axs[j].axes.xaxis.set_visible(False)
            axs[j].axes.yaxis.set_visible(False)

    plt.savefig(f"plots/5_comparison_bary.png")
    plt.close()

    n_plots = len(info_names)
    figsize = (7, 4 * n_plots)
    fig, axs = plt.subplots(n_plots, 1, figsize=figsize)
    for i, names in enumerate(info_names):
        ax = axs[i]
        for name, idx in names.items():
            for trajectory, plot_name in zip(trajs, plot_names):
                ax.plot([el[idx] for el in trajectory[1:]], label=plot_name)
            ax.set_title(name)
        ax.legend()
        ax.yaxis.grid()

    plt.tight_layout()
    plt.savefig(f"plots/5_comparison_convergence.png", bbox_inches='tight')


def plot3d(X, Y, Zs, kappas, sample_size):
    fig = plt.figure(figsize=plt.figaspect(0.33))
    # ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(X, Y)

    def log_tick_formatter_x(val, pos=None):
        return f"{int(2**val)}"

    def log_tick_formatter_y(val, pos=None):
        return f"{round(0.9**val, 2)}"

    # Plot the first surface.
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    axes = [ax1, ax2, ax3]
    for i, ax in enumerate(axes):
        Z = Zs[i]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(0., 1.)
        ax.zaxis.set_major_locator(LinearLocator(6))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter_x))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Initial var')
        ax.set_ylabel('Var decay')

        ax.title.set_text(f"kappa {kappas[i]}")

    fig.suptitle(f'sample_size {sample_size}', fontsize=16)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.savefig(f"plots/dual3d_sample_size_{sample_size}.png")


def get_sampler(sample_size):
    def get_sample(mean, cov, seed):
        if seed is not None:
            torch.manual_seed(seed)
        distr = MultivariateNormal(loc=mean, covariance_matrix=cov)
        sample = distr.sample((sample_size,))
        return sample
    return get_sample


def get_factor(decay, var_decay, step):
    if decay == 'exp':
        return var_decay ** step
    elif decay == 'lin':
        return var_decay / (step + var_decay)
    else:
        return var_decay / (step + np.sqrt(var_decay))


def get_empir_cov(sample, step, weights, decay, var_decay):
    matrix = torch.cov(sample.T, aweights=weights)
    print(f'Cov matrix norm: {torch.linalg.matrix_norm(matrix, 2)}')
    # diag = torch.min(torch.diag(matrix))
    factor = get_factor(decay, var_decay, step)
    return factor * matrix  # (matrix / diag)


def scale_cov(step, decay, var_decay, prior_cov):
    factor = get_factor(decay, var_decay, step)
    return factor * prior_cov


def load_data(m, src_digit, target_digit, device, noise=None, dtype=torch.float32):
    digits = load_digits()

    cs = []
    r_prior = None
    i = 0
    while len(cs) < m:
        digit = digits.target[i]
        is_prior = r_prior is None and digit == src_digit
        if is_prior or digit == target_digit:
            img = torch.from_numpy(digits.data[i]).type(dtype)
            if is_prior and noise is not None:
                img += noise * torch.rand(*img.shape)
            img /= img.sum()

            # plt.figure()
            # plt.imshow(im3.reshape(8, -1), cmap='binary')
            if is_prior:
                r_prior = img
                # plt.savefig(f"r0.png")
            else:
                cs.append(img)
                # plt.savefig(f"c{len(cs)}.png")
        i += 1
    return r_prior.to(device), torch.stack(cs).to(device)


def load_mnist(m, target_digit, device, size=(14, 14), dtype=torch.float32, seed=None):
    transform = Compose([
        Resize(size),
        ToTensor()
    ])
    mnist = MNIST('.', train=False, transform=transform, download=True)
    indexes = (mnist.targets == target_digit).nonzero().flatten().tolist()
    if seed:
        torch.random.seed(seed)
    chosen = random.sample(indexes, m)
    cs = [mnist[i][0] for i in chosen]
    cs = torch.stack(cs).reshape(m, -1).type(dtype)
    return (cs / cs.sum(dim=-1, keepdims=True)).to(device)


def get_sample_generator(prior_mean, n_batches, prior_std, verbose=False):
    def sample_generator():
        for i in range(n_batches):
            if verbose:
                print(f"sampling batch {i}")
            torch.manual_seed(i)
            yield torch.normal(prior_mean, prior_std)

    return sample_generator


def replace_zeros(arr, replace_val=1e-9, sumdim=-1):
    arr[arr < replace_val] = replace_val
    arr /= arr.sum(dim=sumdim, keepdim=True)
    return arr


def test_mnist():
    cs = load_mnist(5, 2, 'cpu', size=(14, 14))
    images = [c for c in cs]
    img_sz = 14
    image_name = 'mnist_test'
    show_barycenters(images, img_sz, image_name, use_softmax=False, scaling='partial')


if __name__ == '__main__':
    # file_names = [f'trajectory_5_samples_512_var_16.0_decay_0.8']\
    #            + [f'trajectory_5_samples_{8 ** i}_var_4.0_decay_1.0' for i in range(4, 6)]
    # plot_names = [f'{8 ** i} samples' for i in range(3, 6)]
    # info_names = [{'Objective': 1}, {'Squared error': 2}]
    # compare_trajectories(file_names, plot_names, info_names, 6, 5)
    test_mnist()
