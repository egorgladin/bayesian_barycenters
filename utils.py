"""
Auxiliary functions.
"""
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter, MaxNLocator

import pickle
from sklearn.datasets import load_digits
from torch.distributions.multivariate_normal import MultivariateNormal


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


def show_barycenters(barycenters, img_sz, img_name, iterations=None):
    """Display several barycenters across iterations."""
    fig, ax = plt.subplots(nrows=1, ncols=len(barycenters), figsize=(16, 4))
    for i, z in enumerate(barycenters):
        img = torch.softmax(z, dim=-1).cpu().numpy().reshape(img_sz, -1)
        ax[i].imshow(img, cmap='binary')
        plt.xticks([])
        plt.yticks([])

    if iterations is not None:
        for i, iter in enumerate(iterations):
            ax[i].title.set_text(f"Iteration {iter}")

    # plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.savefig(f"plots/bary_{img_name}.png", bbox_inches='tight')
    plt.close()


def plot_convergence(trajectory, img_name, info_names, log_scale=False, opt_val=None):
    """Plot info stored in trajectory."""
    n_plots = len(info_names)
    figsize = (7, 4 * n_plots)
    fig, axs = plt.subplots(n_plots, 1, figsize=figsize)
    for i, names in enumerate(info_names):
        ax = axs[i]
        for name, idx in names.items():
            ax.plot([el[idx].to('cpu').numpy() if isinstance(el[idx], torch.Tensor) else el[idx]
                     for el in trajectory[1:]], label=name)
            if name == 'Objective' and opt_val is not None:
                if isinstance(opt_val, torch.Tensor):
                    opt_val = opt_val.to('cpu').numpy()
                ax.hlines(y=opt_val, xmin=0, xmax=len(trajectory)-2, label='Optimal value')
        ax.yaxis.grid()

        ax.legend()
        if log_scale:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"plots/convergence_{img_name}.png", bbox_inches='tight')


def plot_trajectory(trajectory, n_cols, img_sz, img_name, info_names, log_scale=False, opt_val=None):
    """Display several barycenters and plot other info stored in trajectory."""
    with open(f'pickled/trajectory_{img_name}.pickle', 'wb') as handle:
        pickle.dump(trajectory, handle)

    n_steps = len(trajectory) - 1
    # plot_every = int(n_steps / n_cols) + 1 if n_steps % n_cols else int(n_steps / n_cols)
    slope = n_steps / (n_cols - 1)
    iterations = [int(i * slope) for i in range(n_cols)]
    barycenters = [trajectory[it][0].to('cpu') for it in iterations] if info_names is not None\
        else [trajectory[it].to('cpu') for it in iterations]
    show_barycenters(barycenters, img_sz, img_name, iterations=iterations)
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


def load_data(m, src_digit, target_digit, device, noise=None):
    digits = load_digits()

    cs = []
    r_prior = None
    i = 0
    while len(cs) < m:
        digit = digits.target[i]
        is_prior = r_prior is None and digit == src_digit
        if is_prior or digit == target_digit:
            img = torch.from_numpy(digits.data[i].astype('float32'))
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


def replace_zeros(arr, replace_val=1e-9, sumdim=-1):
    arr[arr == 0] = replace_val
    arr /= arr.sum(dim=sumdim, keepdim=True)
    return arr


if __name__ == '__main__':
    file_names = [f'trajectory_5_samples_512_var_16.0_decay_0.8']\
               + [f'trajectory_5_samples_{8 ** i}_var_4.0_decay_1.0' for i in range(4, 6)]
    plot_names = [f'{8 ** i} samples' for i in range(3, 6)]
    info_names = [{'Objective': 1}, {'Squared error': 2}]
    compare_trajectories(file_names, plot_names, info_names, 6, 5)
