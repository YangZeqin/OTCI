import torch

def calculate_upsilon(C, T, gamma, eta, eps, M_c=None, M_t=None, alpha=None, M_cc=None):
    T_i = torch.sum(T, axis=1)

    # Gradient of W，add 1e-5 to avoid log(0)
    gradient = alpha * C + gamma * torch.log(torch.unsqueeze(T_i, axis=1).expand(-1, C.shape[1]) + eps)

    # Gradient of GW
    if M_c is not None and M_t is not None and alpha is not None and M_cc is not None:
        n_t = T.shape[1]
        one_mat = torch.ones([n_t, n_t]).cuda()
        gradient += (1 - alpha) * (2 * torch.mm(torch.mm(M_cc, T), one_mat) - 4 * torch.mm(torch.mm(M_c, T), M_t.T))

    upsilon = T * torch.exp(-eta * gradient)

    return upsilon


def optimal_transport_weighting(X_t, X_c, Y_t, Y_c, gamma=0.1, eta_base=0.01, eta_update_steps=10,
                                decay_rate=0.95, max_iter=2000, abstol=1e-5, eps=1e-3, with_GW=True, alpha=0.1):
    # ================================== 1. initialize ==================================
    n_t, n_c = X_t.shape[0], X_c.shape[0]
    T = (torch.ones([n_c, n_t]) / (n_c * n_t)).cuda()

    # initialize C
    C = (torch.zeros([n_c, n_t])).cuda()
    for row in range(n_c):
        C[row] = torch.sum(torch.square(X_c[row, :] - X_t), axis=1)

    # initialize M_c, M_t
    if with_GW == True:
        M_c = torch.zeros([n_c, n_c]).cuda()
        for row in range(n_c):
            M_c[row] = torch.mm(X_c[row, :].view(1, -1), X_c.T)

        M_t = torch.zeros([n_t, n_t]).cuda()
        for row in range(n_t):
            M_t[row] = torch.mm(X_t[row, :].view(1, -1), X_t.T)

        M_cc = M_c * M_c

    # ================================== 2. training ==================================
    T_old = T
    for i in range(max_iter):
        # 2.1 calculate upsilon
        eta = eta_base * (decay_rate ** (i // eta_update_steps))
        if with_GW == True:
            upsilon = calculate_upsilon(C=C, T=T_old, gamma=gamma, eta=eta, eps=eps, M_c=M_c, M_t=M_t, M_cc=M_cc, alpha=alpha)
        else:
            upsilon = calculate_upsilon(C=C, T=T_old, gamma=gamma, eta=eta, eps=eps)

        # 2.2 update T
        T_j = torch.sum(upsilon, axis=0)
        T_new = upsilon / (n_t * torch.unsqueeze(T_j, axis=0).expand(n_c, -1))

        # 2.3 check early stop
        current_abstol = torch.sum(torch.abs(T_new - T_old))
        T_old = T_new

        if current_abstol < abstol:
            break

    # ================================== 3. calculate weight ==================================
    w = torch.sum(T_new, axis=1)
    pred_att = torch.sum(Y_t) / n_t - torch.sum(w * Y_c)
    return pred_att, w, T_new

