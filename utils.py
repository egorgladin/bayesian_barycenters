import torch
import matplotlib.pyplot as plt


def safe_log(arr, device, minus_inf=-100.):
    return torch.where(arr > 0, torch.log(arr), torch.tensor(minus_inf, device=device))


def plot_trajectory(trajectory, n_cols, img_sz, img_name):
    n_steps = len(trajectory)
    plot_every = int(n_steps / n_cols) + 1 if n_steps % n_cols else int(n_steps / n_cols)
    points_to_plot = trajectory[::plot_every]
    fig, ax = plt.subplots(nrows=1, ncols=len(points_to_plot), figsize=(16, 4))
    for i, z in enumerate(points_to_plot):
        img = torch.softmax(z, dim=-1).cpu().numpy().reshape(img_sz, -1)
        ax[i].imshow(img, cmap='binary')

    plt.savefig(f"plots/{img_name}.png", bbox_inches='tight')
    plt.close()
