import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union
import time

from utils import show_barycenters, get_digits_and_bary, plot_convergence, get_cost_mat, plot_trajectory
from kantorovich_dual import objective_function


def run_grad_descent(n_steps, cost_mat, cs, z_prior, step=0.5, return_what=None, kappa2=None, momentum=0,
                     verbose=False, weight_decay=0):
    if return_what is None:
        return_what = ['val']
    kappa = 1. / 30

    m = cs.shape[0]
    objective_output = ['objective'] + (['barycenter'] if 'barycenter' in return_what else [])
    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa, kappa2=kappa2, inverse_order=True,
                                                  return_what=objective_output, double_conjugate=False)
    L = m / kappa
    opt = torch.optim.SGD([z_prior], lr=step/L, momentum=momentum, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(opt, patience=2, threshold=1e-3, verbose=verbose)

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
            # barys.append(z_prior.clone().detach())
        if verbose and (i % 100 == 0):
            print(f"    GD step {i}/{n_steps}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 200 == 0:
            scheduler.step(loss)

    if len(return_what) == 1 and 'val' in return_what:
        last_vals = n_steps//50
        return sum(losses[-last_vals:]) / last_vals

    res = [best_potential] if 'poten' in return_what else []
    res += [losses] if 'losses' in return_what else []
    res += [barys] if 'barycenter' in return_what else []
    return res


def get_mid(a, b, func):
    start = time.time()
    mid = 0.5 * (a + b)
    f_mid = func(mid)
    print(f"    Parameter: {mid}, objective: {f_mid:.3f} | {time.time() - start:.2f}s")
    return mid, f_mid


def bisection(a, b, n_bisections, func):
    mid, f_mid = get_mid(a, b, func)
    for i in range(n_bisections):
        print(f"Bisection {i}/{n_bisections}")
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


def main(device='cpu', n_steps=1000, init_poten=None, search_stepsize=False, stepsize_bounds=(10, 50), n_bisec=10, stepsize=None,
         momentum: Union[int, float] = 0, mnist=False, weight_decay: Union[int, float] = 0, verbose=True):
    folder = 'digit_experiment/'
    data_fname = '35_images_of_5.pt'
    bary_fname = 'bary_35_im_of_5.pt'

    if mnist:
        img_size = 28
        data_fname = 'mnist_' + data_fname
        bary_fname = 'mnist_' + bary_fname
    else:
        img_size = 8

    dtype = torch.float64
    cost_mat = get_cost_mat(img_size, device, dtype=dtype)

    cs, r = get_digits_and_bary(folder + data_fname, folder + bary_fname, target_digit=5, n_data_points=35,
                                dtype=dtype, device=device, cost_mat=cost_mat, mnist=mnist, verbose=verbose)

    if init_poten:
        z_prior = torch.load(init_poten, map_location=device).requires_grad_()
        if verbose:
            print(f"Loaded starting point from {init_poten}")
    else:
        torch.manual_seed(0)
        z_prior = torch.randn((1, cs.numel()), device=device, dtype=dtype, requires_grad=True)
        if verbose:
            print("Initialized a random starting point")

    if search_stepsize:
        func = lambda step: run_grad_descent(n_steps, cost_mat, cs, z_prior, step=step, momentum=momentum,
                                             weight_decay=weight_decay)
        best_val, best_step = bisection(*stepsize_bounds, n_bisec, func)
        print(f"Best step: {best_step}, best val: {best_val:.2f}")
    else:
        best_step = stepsize

    return_what = ['losses', 'barycenter']
    res = run_grad_descent(n_steps, cost_mat, cs, z_prior, step=best_step, momentum=momentum,
                                           return_what=return_what, kappa2=.1, verbose=verbose,
                                           weight_decay=weight_decay)

    losses, barycenters = res[-2], res[-1]
    barys_errors = [torch.norm(bar - r) for bar in barycenters]

    info_names = [{'loss': 1}, {'Accuracy of r': 2}]
    trajectory = list(zip(barycenters, losses, barys_errors))
    plot_trajectory(trajectory, 10, img_size, f'minus_sgd_step_{best_step}', info_names, log_scale=(False, True),
                    use_softmax=False)


if __name__ == '__main__':
    # main('cuda', stepsize=13.9453125, momentum=0.9, plot_traj=False)
    # for stepsize in [10.**i for i in range(1, 3)]:
    #     main('cuda',
    #          n_steps=2500,
    #          stepsize=stepsize,
    #          momentum=0.9,
    #          mnist=True)
    main('cuda',
         n_steps=5000,
         stepsize=10.,
         momentum=0.9,
         mnist=True)
    # main('cuda',
    #      n_steps=2500,
    #      init_poten='digit_experiment/starting_point.pt',
    #      search_stepsize=True,
    #      stepsize_bounds=(1, 1000),
    #      n_bisec=4,
    #      momentum=0.9,
    #      weight_decay=1e-4,
    #      mnist=True)
