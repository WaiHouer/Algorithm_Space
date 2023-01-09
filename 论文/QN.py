"""
拟牛顿法
"""
import time

import numpy as np
from sympy import symbols, diff
from GAQN_Func import GAQN_Func


def QN(K, S, E, A, U, N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_q, delta_u
       , gamma_a, gamma_q, gamma_u, p, q, eta, b_hat, lambda_b, C, lambda_c, b_init, c_init
       , B, b_last):
    for k in range(K):
        b_init[k][0], b_init[k][1] = 70, 140
        c_init[k][0], c_init[k][1] = 50000, 50000

    f, S1, E1, A1, U1 = GAQN_Func(K, S, E, A, U, N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_u
                                  , gamma_a, gamma_u, p, q, eta)  # 目标函数表达式（两期）
    x = [symbols(f'x{i}') for i in range(36)]
    g = gradient(x, f)  # 一阶导数的表达式

    x_init = np.zeros(36)  # 初始解
    x_init[0:K] = b_init[:, 0]
    x_init[K:2 * K] = c_init[:, 0]
    x_init[2 * K:3 * K] = b_init[:, 1]
    x_init[3 * K:4 * K] = c_init[:, 1]

    for i in range(5):
        print(f'第{i}次')

        g_tem = np.array([None for i in range(36)])
        g_tem[:] = g[:]
        print('g_tem', g_tem[0])
        for j in range(36):  # 一阶导数
            for k in range(36):
                g_tem[j] = g_tem[j].subs(x[k], x_init[k])
        g_tem = g_tem.astype(float)

        if np.linalg.norm(g_tem) <= 0.1:
            print('ggg111')
            break
        p = - g_tem  # 下降方向

        x_new = lam(p, x_init, K, B, b_last, b_hat, lambda_b, C, lambda_c, f, x, U1, A1, eta, U, A, N)

        # if x_new.all() == x_init.all():
        #     print('gggg2222')
        #     break

        x_init = x_new


def gradient(x, func):
    g = np.array([None for i in range(36)])
    for i in range(36):
        g[i] = diff(func, x[i])  # 求导，形成表达式
    return g


def lam(p, x_init, K, B, b_last, b_hat, lambda_b, C, lambda_c, f, x_v, U1, A1, eta, U, A, N):
    lam = 1
    tag = 0
    x = x_init
    f_tem = f
    for i in range(36):
        f_tem = f_tem.subs(x_v[i], x[i])
    print(f_tem)
    while True:
        x_new = x + lam * p
        if sum(x_new[0:K]) > B[0] or sum(x_new[2 * K:3 * K]) > B[1]:
            print('1')
            tag = 1

        for k in range(K):
            if x_new[k] < b_last[k] or x_new[k + 2 * K] < x_new[k]:
                tag = 1
                print('2')

            if x_new[k] - b_last[k] > lambda_b * b_hat[0] or x_new[k + 2 * K] - x_new[k] > lambda_b * b_hat[1]:
                tag = 1
                print('3')

            U_tem, A_tem = U1[k], A1[k]
            for i in range(36):
                U_tem, A_tem = U_tem.subs(x_v[i], x_new[i]), A_tem.subs(x_v[i], x_new[i])
            if eta * x_new[k] > U[k] + A[k] / N[k] * x_new[k + K]:
                tag = 1
            if eta * x_new[k + 2 * K] > U_tem + A_tem / N[k] * x_new[k + 3 * K]:
                tag = 1

        if sum(x_new[K:2 * K]) > C[0] or sum(x_new[3 * K:4 * K]) > C[1]:
            tag = 1

        for k in range(K):
            if x_new[k + K] > N[k] or x_new[k + 3 * K] > N[k]:
                tag = 1

            if x_new[k + K] > lambda_c * C[0] or x_new[k + 3 * K] > lambda_c * C[1]:
                tag = 1

        if tag == 1:
            break
        else:
            x = x_new

    f_tem = f
    for i in range(36):
        f_tem = f_tem.subs(x_v[i], x[i])
    print(f_tem)
    return x


