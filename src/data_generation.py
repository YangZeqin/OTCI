import random
import pandas as pd
import numpy as np

def generate_multi_normal_data(feat_dim=10, mut=0.5, boundt=0.5, muc=1.0, boundc=1.0, numt=1500, numc=1500, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # --------------------- X --------------------------
    # Treatment group samples
    mu_treat= np.zeros(feat_dim) + mut
    bound_treat = boundt
    cov_marix_treat = np.random.uniform(low=0, high=bound_treat, size=(feat_dim, feat_dim))
    Sigma_treat = 0.5 * (np.dot(cov_marix_treat, cov_marix_treat.transpose()))
    X_treat = np.random.multivariate_normal(mean=mu_treat, cov=Sigma_treat, size=numt)

    # Control group samples
    mu_control = np.zeros(feat_dim) + muc
    bound_control = boundc
    cov_marix_control = np.random.uniform(low=0, high=bound_control, size=(feat_dim, feat_dim))
    Sigma_control = 0.5 * (np.dot(cov_marix_control, cov_marix_control.transpose()))
    X_control = np.random.multivariate_normal(mean=mu_control, cov=Sigma_control, size=numc)

    # --------------------- augument X ---------------------
    pow_X_treat = X_treat ** 2
    pow_X_control = X_control ** 2

    # --------------------- Y --------------------------
    W1 = np.random.uniform(low=0, high=1, size=X_treat.shape[1])
    W2 = np.random.uniform(low=0, high=1, size=pow_X_treat.shape[1])
    eps = np.random.normal(loc=0, scale=0.1, size=numt + numc)
    Y_treat = 1 + np.sin(np.dot(X_treat, W1)) + np.cos(np.dot(pow_X_treat, W2)) + eps[:numt]
    Y_control = np.sin(np.dot(X_control, W1)) + np.cos(np.dot(pow_X_control, W2)) + eps[numt:]

    # --------------------- concate ---------------------
    T_treat = np.ones(numt)
    T_control = np.zeros(numc)

    data_treat = np.concatenate((T_treat.reshape(-1, 1), X_treat, Y_treat.reshape(-1, 1)), axis=1)
    data_control = np.concatenate((T_control.reshape(-1, 1), X_control, Y_control.reshape(-1, 1)), axis=1)

    df = pd.DataFrame(np.concatenate((data_treat, data_control), axis=0))
    df.columns = ["t",] + ["x_" + str(int(i)) for i in range(1, X_treat.shape[1] + 1)] + ["y",]

    return df



if __name__ == "__main__":
    for seed in range(10):
        df = generate_multi_normal_data(seed=seed, linear=False, muc=1.0, boundc=1.0, feat_dim=10)
        pass