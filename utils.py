import torch
import matplotlib.pyplot as plt
import pickle


def safe_log(arr, device, minus_inf=-100.):
    """Elementwise logarithm with log(0) defined by minus_inf."""
    return torch.where(arr > 0, torch.log(arr), torch.tensor(minus_inf, device=device))


def norm_sq(tensor):
    """Squared Euclidean norm."""
    return (tensor ** 2).sum().item()


def show_barycenters(barycenters, img_sz, img_name):
    fig, ax = plt.subplots(nrows=1, ncols=len(barycenters), figsize=(16, 4))
    for i, z in enumerate(barycenters):
        img = torch.softmax(z, dim=-1).cpu().numpy().reshape(img_sz, -1)
        ax[i].imshow(img, cmap='binary')

    plt.savefig(f"plots/bary_{img_name}.png", bbox_inches='tight')
    plt.close()


def plot_convergence(trajectory, img_name):
    all_names = [{'residue_r': 1, 'residue_c': 2, 'objective': 3, 'transport_cost': 4}, {'acc_X1': 5, 'acc_X2': 6},
                 {'acc_r': 7}]
    fig, axs = plt.subplots(len(all_names), 1, figsize=(7, 15))
    for i, names in enumerate(all_names):
        ax = axs[i]
        for name, idx in names.items():
            ax.plot([el[idx] for el in trajectory[1:]], label=name)

        ax.legend()
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"plots/convergence_{img_name}.png", bbox_inches='tight')


def plot_trajectory(trajectory, n_cols, img_sz, img_name):
    with open(f'trajectory_{img_name}.pickle', 'wb') as handle:
        pickle.dump(trajectory, handle)

    n_steps = len(trajectory)
    plot_every = int(n_steps / n_cols) + 1 if n_steps % n_cols else int(n_steps / n_cols)
    barycenters = [el[0] for el in trajectory[::plot_every]]
    show_barycenters(barycenters, img_sz, img_name)
    plot_convergence(trajectory, img_name)
