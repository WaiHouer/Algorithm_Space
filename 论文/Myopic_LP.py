"""
短视迭代优化算法——Myopic LP
"""
import numpy as np
from Myopic_Model import myopic_model
from Allocation_Epidemic_Function import allocation_epidemic_function


def myopic_lp(K, T, S_initial, E_initial, A_initial, Q_initial, U_initial, R_initial, D_initial, N
              , sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_q, delta_u
              , gamma_a, gamma_q, gamma_u, p, q
              , eta, b_hat, lambda_b, C, lambda_c
              , re_tag=None):
    # 初始化决策向量、目标函数值
    b, c = np.zeros((K, T + 1)), np.zeros((K, T + 1))
    value = np.zeros(T + 1)
    # T+1的目的：把起点放进来
    S, E, A, Q, U, R, D = np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)) \
        , np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))

    for k in range(K):  # 起点初始化
        S[k][0] = S_initial[k]
        E[k][0] = E_initial[k]
        A[k][0] = A_initial[k]
        Q[k][0] = Q_initial[k]
        U[k][0] = U_initial[k]
        R[k][0] = R_initial[k]
        D[k][0] = D_initial[k]

    for t in range(T + 1):
        S_tem = S[:, t]  # 当期状态
        E_tem = E[:, t]
        A_tem = A[:, t]
        U_tem = U[:, t]

        Q_tem = Q[:, t]
        R_tem = R[:, t]
        D_tem = D[:, t]
        if t == 0:
            b_last = np.zeros(K)  # 上期病床
        else:
            b_last = b[:, t - 1]

        b[:, t], c[:, t], value[t] = myopic_model(K, S_tem, E_tem, A_tem, U_tem, b_last, N, sigma_hat, beta_e, beta_a, beta_u
                                                  , eta, b_hat[0: t + 1], lambda_b, C[t], lambda_c)

        # 注意：这里必须用b[:, t:t+1]，而不用b[:, t]，两者虽然在所包含元素上没有任何区别，但是👇
        # b[:, t:t+1]拥有双重下标[k][t]，b[:, t]只拥有一个下标[k]，前者符合allocation_epidemic_function函数对b的要求
        S_nxt, E_nxt, A_nxt, Q_nxt, U_nxt, R_nxt, D_nxt = allocation_epidemic_function(K, 1, S_tem, E_tem, A_tem, Q_tem
                                                                                       , U_tem, R_tem, D_tem, N, sigma_hat
                                                                                       , beta_e, beta_a, beta_u, alpha
                                                                                       , delta_a, delta_q, delta_u
                                                                                       , gamma_a, gamma_q, gamma_u
                                                                                       , p, q, b[:, t:t+1], c[:, t:t+1]
                                                                                       , eta)
        if t != T:  # 更新下一期
            S[:, t + 1] = S_nxt[:, 1]  # S_nxt第一列是本期，第二列是下一期
            E[:, t + 1] = E_nxt[:, 1]
            A[:, t + 1] = A_nxt[:, 1]
            Q[:, t + 1] = Q_nxt[:, 1]
            U[:, t + 1] = U_nxt[:, 1]
            R[:, t + 1] = R_nxt[:, 1]
            D[:, t + 1] = D_nxt[:, 1]

    # for t in range(10):  # 验证求解是否正确，目标函数是否写错了
    #     tem = 0
    #     for k in range(K):
    #         tem += E[k][t+1] - E[k][t] + alpha[k] * E[k][t]
    #     print(tem, value[t])

    if re_tag:
        return b, c, value, S, E, A, U
    else:
        return b, c, value
