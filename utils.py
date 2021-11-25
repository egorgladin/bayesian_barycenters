import torch
import matplotlib.pyplot as plt


def safe_log(arr, device, minus_inf=-100.):
    return torch.where(arr > 0, torch.log(arr), torch.tensor(minus_inf, device=device))


def norm_sq(tensor):
    return (tensor ** 2).sum().item()


def show_barycenters(barycenters, img_sz, img_name):
    fig, ax = plt.subplots(nrows=1, ncols=len(barycenters), figsize=(16, 4))
    for i, z in enumerate(barycenters):
        img = torch.softmax(z, dim=-1).cpu().numpy().reshape(img_sz, -1)
        ax[i].imshow(img, cmap='binary')

    plt.savefig(f"plots/bary_{img_name}.png", bbox_inches='tight')
    plt.close()


def plot_convergence(trajectory, img_name):
    fig, axs = plt.subplots(3, 1, figsize=(7, 15))
    all_names = [{'residue_r': 1, 'residue_c': 2}, {'acc_X1': 3, 'acc_X2': 4}, 'acc_r']
    for ax, names in zip(axs, all_names):
        if isinstance(names, dict):
            for name, idx in names.items():
                pts_to_plot = [el[idx] for el in trajectory[1:]] if idx in [1, 2]\
                    else [el[idx] for el in trajectory[1:]]
                ax.plot(pts_to_plot, label=name)
        else:
            ax.plot([el[5] for el in trajectory], label=names)

        ax.legend()
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"plots/convergence_{img_name}.png", bbox_inches='tight')


def plot_trajectory(trajectory, n_cols, img_sz, img_name):  # r, residue_r, residue_c, acc_X1, acc_X2, acc_r
    n_steps = len(trajectory)
    plot_every = int(n_steps / n_cols) + 1 if n_steps % n_cols else int(n_steps / n_cols)
    barycenters = [el[0] for el in trajectory[::plot_every]]
    show_barycenters(barycenters, img_sz, img_name)
    plot_convergence(trajectory, img_name)
