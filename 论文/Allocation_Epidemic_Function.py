"""
考虑资源分配的传染病模型（疫苗+病床）
"""
from numba import njit
import numpy as np


@njit()
def allocation_epidemic_function(K, T, S_initial, E_initial, A_initial, Q_initial, U_initial, R_initial, D_initial, N
                                 , sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_q, delta_u
                                 , gamma_a, gamma_q, gamma_u, p, q
                                 , b, c, eta):
    # T+1的目的：把起点放进来
    S, E, A, Q, U, R, D = np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))\
        , np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))

    for k in range(K):  # 起点初始化（由于加速器，循环形式比下面的形式更快）
        S[k][0] = S_initial[k]
        E[k][0] = E_initial[k]
        A[k][0] = A_initial[k]
        Q[k][0] = Q_initial[k]
        U[k][0] = U_initial[k]
        R[k][0] = R_initial[k]
        D[k][0] = D_initial[k]

    for k in range(K):
        for t in range(1, T + 1):
            e_trans = 0
            a_trans = 0
            u_trans = 0
            for j in range(K):
                e_trans += E[j][t - 1] / N[j] * beta_e[j] * sigma_hat[k][j]
                a_trans += (A[j][t - 1] - A[j][t - 1] / N[j] * c[j][t - 1])\
                    / N[j] * beta_a[j] * sigma_hat[k][j]
                u_trans += (U[j][t - 1] + A[j][t - 1] / N[j] * c[j][t - 1] - eta * b[j][t - 1])\
                    / N[j] * beta_u[j] * sigma_hat[k][j]
            # 迭代开始：
            S[k][t] = S[k][t - 1] - S[k][t - 1] * (e_trans + a_trans + u_trans)
            E[k][t] = E[k][t - 1] + S[k][t - 1] * (e_trans + a_trans + u_trans) - alpha[k] * E[k][t - 1]
            A[k][t] = (A[k][t - 1] - A[k][t - 1] / N[k] * c[k][t - 1]) + p * alpha[k] * E[k][t - 1] \
                - delta_a[k] * (A[k][t - 1] - A[k][t - 1] / N[k] * c[k][t - 1]) \
                - gamma_a[k] * (A[k][t - 1] - A[k][t - 1] / N[k] * c[k][t - 1])
            Q[k][t] = (Q[k][t - 1] + eta * b[k][t - 1]) + (1 - p) * q * alpha[k] * E[k][t - 1] \
                - delta_q[k] * (Q[k][t - 1] + eta * b[k][t - 1]) \
                - gamma_q[k] * (Q[k][t - 1] + eta * b[k][t - 1])
            U[k][t] = (U[k][t - 1] + A[k][t - 1] / N[k] * c[k][t - 1] - eta * b[k][t - 1]) \
                + (1 - p) * (1 - q) * alpha[k] * E[k][t - 1] \
                - delta_u[k] * (U[k][t - 1] + A[k][t - 1] / N[k] * c[k][t - 1] - eta * b[k][t - 1]) \
                - gamma_u[k] * (U[k][t - 1] + A[k][t - 1] / N[k] * c[k][t - 1] - eta * b[k][t - 1])
            R[k][t] = R[k][t - 1] \
                + gamma_a[k] * (A[k][t - 1] - A[k][t - 1] / N[k] * c[k][t - 1]) \
                + gamma_q[k] * (Q[k][t - 1] + eta * b[k][t - 1]) \
                + gamma_u[k] * (U[k][t - 1] + A[k][t - 1] / N[k] * c[k][t - 1] - eta * b[k][t - 1])
            D[k][t] = D[k][t - 1] \
                + delta_a[k] * (A[k][t - 1] - A[k][t - 1] / N[k] * c[k][t - 1]) \
                + delta_q[k] * (Q[k][t - 1] + eta * b[k][t - 1]) \
                + delta_u[k] * (U[k][t - 1] + A[k][t - 1] / N[k] * c[k][t - 1] - eta * b[k][t - 1])
    return S, E, A, Q, U, R, D
