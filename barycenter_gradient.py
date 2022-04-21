import torch
import ot

from utils import load_data, show_barycenters, replace_zeros, get_digits_and_bary, plot_convergence
from experiment_barycenter import get_cost_matrix
from kantorovich_dual import objective_function


def run_sgd(n_steps, cost_mat, cs, dtype, step=0.5, device='cpu', return_what=None, kappa2=None):
    if return_what is None:
        return_what = ['val']
    kappa = 1. / 30
    temp = 30.

    m = cs.shape[0]
    n = cost_mat.shape[0]
    objective_output = ['objective'] + (['barycenter'] if 'barycenter' in return_what else [])
    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa, kappa2=kappa2,
                                                  return_what=objective_output, double_conjugate=False)
    L = m / kappa

    torch.manual_seed(0)
    z_prior = torch.randn((1, m * n), device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.SGD([z_prior], lr=step/L)

    losses = []
    best_loss = 1e6
    best_potential = None
    barys = []

    for i in range(n_steps):
        obj_out = objective(z_prior)
        loss = -obj_out[0]
        losses.append(loss.item())
        if ('poten' in return_what) and losses[-1] < best_loss:
            best_loss = losses[-1]
            best_potential = z_prior.clone().detach()
        if 'barycenter' in return_what:
            barys.append(obj_out[1].detach())
        # if i % 100 == 0:
        #     print(f"Loss: {loss.item():.2f}")
        opt.zero_grad()
        loss.backward()
        opt.step()

    if len(return_what) == 1 and 'val' in return_what:
        ten_persent = n_steps//10
        return sum(losses[-ten_persent:]) / ten_persent

    res = [best_potential] if 'poten' in return_what else []
    res += [losses] if 'losses' in return_what else []
    res += [barys] if 'barycenter' in return_what else []
    return res


def get_mid(a, b, func):
    mid = 0.5 * (a + b)
    return mid, func(mid)


def bisection(a, b, n_bisections, func):
    mid, f_mid = get_mid(a, b, func)
    for i in range(n_bisections):
        left, f_left = get_mid(a, mid, func)
        right, f_right = get_mid(mid, b, func)
        if f_left < min(f_mid, f_right):
            b = mid
            mid, f_mid = left, f_left
        elif f_mid <= min(f_left, f_right):
            a = left
            b = right
        else:
            a = mid
            mid, f_mid = right, f_right

    return min([(f_left, right), (f_mid, mid), (f_right, right)])


def main(device='cpu', search_stepsize=False, plot_traj=True):
    dtype = torch.float64
    img_size = 8
    cost_mat = get_cost_matrix(img_size, device, dtype=dtype)
    folder = 'digit_experiment/'

    data_path = folder + '20_images_of_3.pt'
    bary_path = folder + 'bary_20_im_of_3.pt'

    cs, r = get_digits_and_bary(data_path, bary_path, target_digit=5, n_data_points=20,
                                dtype=torch.float32, device=device, cost_mat=cost_mat)

    n_steps = 1000

    if search_stepsize:
        func = lambda step: run_sgd(n_steps, cost_mat, cs, dtype, step=step, device=device)
        best_val, best_step = bisection(30, 200, 10, func)
        print(f"Best step: {best_step}, best val: {best_val:.2f}")
    else:
        best_step = 77.8125

    if plot_traj:
        losses, barycenters = run_sgd(n_steps, cost_mat, cs, dtype, step=best_step,
                                      device=device, return_what=['losses', 'barycenter'])
        barys_errors = [torch.norm(bar - r) for bar in barycenters]

        info_names = [{'loss': 0}, {'Accuracy of r': 1}]
        plot_convergence(list(zip(losses, barys_errors)), 'losses_barys_errors',
                         info_names, log_scale=True, opt_val=None)

    else:
        best_potential = run_sgd(n_steps, cost_mat, cs, dtype, step=best_step, device=device,
                                 return_what=['poten'], kappa2=.1)[0]

        barys, titles = [r], ['True bary']
        kaps = [.1, .12, .14, .16]
        for kap in kaps:
            barys.append(
                objective_function(best_potential.reshape(1, -1), cost_mat, cs, kap, return_what=['barycenter'])[0]
            )
            titles.append(f'Bary from poten, k={kap}')
        show_barycenters(barys, img_size, 'bary_poten_sgd', use_softmax=False, iterations=titles)


if __name__ == '__main__':
    main('cpu')
