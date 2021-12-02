"""
Auxiliary functions.
"""
import torch
import matplotlib.pyplot as plt
import pickle


def safe_log(arr, minus_inf=-100.):
    """Elementwise logarithm with log(0) defined by minus_inf."""
    minus_inf_tensor = minus_inf * torch.ones_like(arr)
    return torch.where(arr > 0, torch.log(arr), minus_inf_tensor)


def norm_sq(tensor):
    """Squared Euclidean norm."""
    return (tensor ** 2).sum().item()


def show_barycenters(barycenters, img_sz, img_name):
    """Display several barycenters across iterations."""
    fig, ax = plt.subplots(nrows=1, ncols=len(barycenters), figsize=(16, 4))
    for i, z in enumerate(barycenters):
        img = torch.softmax(z, dim=-1).cpu().numpy().reshape(img_sz, -1)
        ax[i].imshow(img, cmap='binary')

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
            ax.plot([el[idx] for el in trajectory[1:]], label=name)
            if name == 'Objective' and opt_val is not None:
                ax.hlines(y=opt_val, xmin=0, xmax=len(trajectory)-1, color='green', label='Optimal value')

        ax.legend()
        if log_scale:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"plots/convergence_{img_name}.png", bbox_inches='tight')


def plot_trajectory(trajectory, n_cols, img_sz, img_name, info_names, log_scale=False, opt_val=None):
    """Display several barycenters and plot other info stored in trajectory."""
    with open(f'trajectory_{img_name}.pickle', 'wb') as handle:
        pickle.dump(trajectory, handle)

    n_steps = len(trajectory)
    plot_every = int(n_steps / n_cols) + 1 if n_steps % n_cols else int(n_steps / n_cols)
    barycenters = [el[0] for el in trajectory[::plot_every]]
    show_barycenters(barycenters, img_sz, img_name)
    plot_convergence(trajectory, img_name, info_names, log_scale=log_scale, opt_val=opt_val)
