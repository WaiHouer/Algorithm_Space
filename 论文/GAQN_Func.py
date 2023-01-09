"""
两期目标函数表达式
"""
from sympy import symbols


def GAQN_Func(K, S, E, A, U, N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_u
              , gamma_a, gamma_u, p, q, eta):
    x = [symbols(f'x{i}') for i in range(36)]

    f_t = 0
    constant_1, constant_2, constant_3 = 0, 0, 0  # 计算常数项
    for k in range(K):
        tem_1, tem_2, tem_3 = 0, 0, 0
        for j in range(K):
            tem_1 += E[j] / N[j] * beta_e[j] * sigma_hat[k][j]
            tem_2 += A[j] / N[j] * beta_a[j] * sigma_hat[k][j]
            tem_3 += U[j] / N[j] * beta_u[j] * sigma_hat[k][j]
        constant_1 += S[k] * tem_1
        constant_2 += S[k] * tem_2
        constant_3 += S[k] * tem_3
    f_t += constant_1 + constant_2 + constant_3

    for j in range(K):  # 计算t期的，含决策变量项
        tem_b, tem_c = 0, 0
        for k in range(K):
            tem_b += S[k] * sigma_hat[k][j]
            tem_c += S[k] * sigma_hat[k][j]
        tem_b = tem_b * eta / N[j] * beta_u[j]
        tem_c = tem_c * A[j] / N[j] / N[j] * (beta_u[j] - beta_a[j])
        f_t -= tem_b * x[j]  # b_jt对应0-8
        f_t += tem_c * x[j + K]  # c_jt对应9-17

    # 用决策变量，迭代算出t+1期的群体
    S1, E1, A1, U1 = [None for i in range(K)], [None for i in range(K)], [None for i in range(K)], [None for i in range(K)]
    for k in range(K):
        tem_S1, tem_E1, tem_A1 = 0, 0, 0
        for j in range(K):
            tem_S1 -= E[j] / N[j] * beta_e[j] * sigma_hat[k][j]
            tem_S1 -= A[j] / N[j] * beta_a[j] * sigma_hat[k][j]
            tem_S1 -= U[j] / N[j] * beta_u[j] * sigma_hat[k][j]
            tem_S1 += eta / N[j] * x[j] * beta_u[j] * sigma_hat[k][j]  # b_jt对应0-8
            tem_S1 -= A[j] / N[j] / N[j] * x[j + K] * (beta_u[j] - beta_a[j]) * sigma_hat[k][j]  # c_jt对应9-17
        S1[k] = S[k] + S[k] * tem_S1
        E1[k] = E[k] * (1 - alpha[k]) - S[k] * tem_S1
        A1[k] = A[k] * (1 - delta_a[k] - gamma_a[k]) + p * alpha[k] * E[k] \
                - (1 - delta_a[k] - gamma_a[k]) * A[k] / N[k] * x[k + K]  # c_jt对应9-17
        U1[k] = U[k] * (1 - delta_u[k] - gamma_u[k]) + (1 - p) * (1 - q) * alpha[k] * E[k] \
                + (1 - delta_u[k] - gamma_u[k]) * A[k] / N[k] * x[k + K] \
                - (1 - delta_u[k] - gamma_u[k]) * eta * x[k]  # b_jt对应0-8, c_jt对应9-17

    f_t1 = 0
    const_t1_1, const_t1_2, const_t1_3 = 0, 0, 0  # 同理t期，计算各项
    for k in range(K):
        tem_1, tem_2, tem_3 = 0, 0, 0
        for j in range(K):
            tem_1 += E1[j] / N[j] * beta_e[j] * sigma_hat[k][j]
            tem_2 += A1[j] / N[j] * beta_a[j] * sigma_hat[k][j]
            tem_3 += U1[j] / N[j] * beta_u[j] * sigma_hat[k][j]
        const_t1_1 += S1[k] * tem_1
        const_t1_2 += S1[k] * tem_2
        const_t1_3 += S1[k] * tem_3
    f_t1 += const_t1_1 + const_t1_2 + const_t1_3

    for j in range(K):
        tem_b, tem_c = 0, 0
        for k in range(K):
            tem_b += S1[k] * sigma_hat[k][j]
            tem_c += S1[k] * sigma_hat[k][j]
        tem_b = tem_b * eta / N[j] * beta_u[j]
        tem_c = tem_c * A1[j] / N[j] / N[j] * (beta_u[j] - beta_a[j])
        f_t1 -= tem_b * x[j + 2 * K]  # b_jt+1对应18-26
        f_t1 += tem_c * x[j + 3 * K]  # c_jt+1对应27-35

    f = f_t + f_t1
    return f, S1, E1, A1, U1
