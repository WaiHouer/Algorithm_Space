"""
自然状态下的迭代
"""
import numpy as np
from Allocation_Epidemic_Function import allocation_epidemic_function


def nature_process(K, T, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, N, sigma_hat, beta_e, beta_a
                   , beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q, eta):
    S, E, A, Q, U, R, D = np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)) \
        , np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))
    b, c = np.zeros((K, T + 1)), np.zeros((K, T + 1))

    for k in range(K):
        S[k][0] = S_init[k]
        E[k][0] = E_init[k]
        A[k][0] = A_init[k]
        Q[k][0] = Q_init[k]
        U[k][0] = U_init[k]
        R[k][0] = R_init[k]
        D[k][0] = D_init[k]

    for t in range(T + 1):
        S_tem = S[:, t]  # 当期状态
        E_tem = E[:, t]
        A_tem = A[:, t]
        U_tem = U[:, t]

        Q_tem = Q[:, t]
        R_tem = R[:, t]
        D_tem = D[:, t]

        S_nxt, E_nxt, A_nxt, Q_nxt, U_nxt, R_nxt, D_nxt = \
            allocation_epidemic_function(K, 1, S_tem, E_tem, A_tem, Q_tem, U_tem
                                         , R_tem, D_tem, N, sigma_hat, beta_e, beta_a
                                         , beta_u, alpha, delta_a, delta_q, delta_u
                                         , gamma_a, gamma_q, gamma_u, p, q
                                         , b[:, t:t + 1], c[:, t:t + 1], eta)

        if t != T:  # 更新下一期
            S[:, t + 1] = S_nxt[:, 1]  # S_nxt第一列是本期，第二列是下一期
            E[:, t + 1] = E_nxt[:, 1]
            A[:, t + 1] = A_nxt[:, 1]
            Q[:, t + 1] = Q_nxt[:, 1]
            U[:, t + 1] = U_nxt[:, 1]
            R[:, t + 1] = R_nxt[:, 1]
            D[:, t + 1] = D_nxt[:, 1]
    return S, E, A, Q, U, R, D
