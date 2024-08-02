import argparse
import numpy as np
import random
import torch
from data_generation import generate_multi_normal_data
from otci import optimal_transport_weighting


def otw(gamma, eta_base, eta_update_steps, decay_rate, abstol, eps, max_iter, seed, with_GW=True, alpha=0.1):

    # read dataset
    data = generate_multi_normal_data(seed=seed, muc=1.0, boundc=1.0, feat_dim=10)
    x_col = ["x_" + str(int(i)) for i in range(1, 11)]
    X_t, X_c = data.loc[data["t"] == 1, x_col].values, data.loc[data["t"] == 0, x_col].values
    Y_t, Y_c = data.loc[data["t"] == 1, "y"].values, data.loc[data["t"] == 0, "y"].values
    gt_att = 1.0

    X_t, X_c = torch.tensor(X_t).cuda(), torch.tensor(X_c).cuda()
    Y_t, Y_c = torch.tensor(Y_t).cuda(), torch.tensor(Y_c).cuda()

    pred_att, w, T_new = optimal_transport_weighting(X_t, X_c, Y_t, Y_c, gamma=gamma, eta_base=eta_base,
                                           eta_update_steps=eta_update_steps, decay_rate=decay_rate,
                                           abstol=abstol, eps=eps, max_iter=max_iter, with_GW=with_GW, alpha=alpha)

    print("seed: {}, True_ATT: {:.5f}, Pred_ATT: {:.5f}, Error: {:.5f}".format(seed, gt_att, pred_att, abs(gt_att - pred_att)))

    return abs(gt_att - pred_att)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=2, help='cuda device')
    parser.add_argument('--dataset', type=str, default="synthetic_data", help='use synthetic_data as running example')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')

    # param related to negative entropy regularization
    parser.add_argument('--gamma', type=float, default=1e-4, help='strength of negative entropy regularization.')
    parser.add_argument('--eps', type=float, default=1e-5, help='avoid log() becoming 0.')

    # param related to learning rate
    parser.add_argument('--eta_base', type=float, default=5e-1, help='learning rate.')
    parser.add_argument('--eta_update_steps', type=int, default=20, help='update frequency of learning rate.')
    parser.add_argument('--decay_rate', type=float, default=0.98, help='decay rate of learning rate.')

    # other param
    parser.add_argument('--abstol', type=float, default=1e-3, help='stop early condition.')
    parser.add_argument('--max_iter', type=int, default=500, help='max_iter.')

    # param related to GW
    parser.add_argument('--with_GW', type=int, default=1, help='if add GW')
    parser.add_argument('--alpha', type=float, default=0.95, help='strength of GW')

    args = parser.parse_args()

    dataset = args.dataset
    gamma = args.gamma
    eta_base = args.eta_base
    eta_update_steps = args.eta_update_steps
    decay_rate = args.decay_rate
    abstol = args.abstol
    eps = args.eps
    max_iter = args.max_iter
    seed = args.seed
    if args.with_GW == 1:
        with_GW = True
    else:
        with_GW = False
    alpha = args.alpha
    device = args.device

    error_list = []

    torch.cuda.set_device(device)

    print("Running Simulation Data(Gaussian, mu_c=1.0) as example!")
    for seed in range(10):
        random.seed(seed)
        np.random.seed(seed)

        err = otw(gamma=gamma, eta_base=eta_base, eta_update_steps=eta_update_steps, decay_rate=decay_rate, abstol=abstol,
                  eps=eps, max_iter=max_iter, seed=seed, with_GW=with_GW, alpha=alpha)
        error_list.append(err.item())

    mae = np.array(error_list).mean()
    std = np.array(error_list).std()

    print()
    print("gamma: {}, eta_base: {}, alpha: {}, Mean_MAE: {:.5f}".format(gamma, eta_base, alpha, mae))
    print("gamma: {}, eta_base: {}, alpha: {}, Mean_STD: {:.5f}".format(gamma, eta_base, alpha, std))


