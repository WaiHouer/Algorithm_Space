"""
ADP算法
"""
import random
import time

import numpy as np
from Allocation_Epidemic_Function import allocation_epidemic_function
import math
import random
from Myopic_Model import myopic_model
from Benchmark import benchmark
from ADP_Model import adp_model
from ADP_FNN import adp_fnn


def adp(K, T, S_initial, E_initial, A_initial, Q_initial, U_initial, R_initial, D_initial
        , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_q, delta_u
        , gamma_a, gamma_q, gamma_u, p, q, eta, b_hat, C, E_initial_last
        , M, L, lambda_b, lambda_c, W):
    # 初始化决策向量、目标函数值
    b, c = np.zeros((K, T + 1)), np.zeros((K, T + 1))
    value = np.zeros(T + 1)
    # T+1的目的：把起点放进来
    S, E, A, Q, U, R, D = np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)) \
        , np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))

    h = np.zeros((T + 1, M, L, 5 * K))  # 初始化状态矩阵
    hh = np.zeros((T + 1, M, L, 3 * K))  # 用于存放状态Q,R,D，只有在引用启发式时用得到
    b_before_time = np.zeros(K)  # 初始化时间跨度前一期的决策0
    theta_1 = np.zeros((T + 1, 5 * K, W))  # 隐藏层权值矩阵
    phi_1 = np.zeros((T + 1, W))  # 隐藏层偏置向量
    theta_2 = np.zeros((T + 1, W))  # 输出层权值向量
    phi_2 = np.zeros(T + 1)  # 输出层偏置值

    for k in range(K):  # 起点初始化
        S[k][0] = S_initial[k]
        E[k][0] = E_initial[k]
        A[k][0] = A_initial[k]
        Q[k][0] = Q_initial[k]
        U[k][0] = U_initial[k]
        R[k][0] = R_initial[k]
        D[k][0] = D_initial[k]

    for m in range(M):
        s_m_time = time.time()
        print(f'第{m + 1}次修正，开始')
        s_s_time = time.time()
        for t in range(T):  # （1）正向采样
            for l in range(L):
                if t == 0:  # 初始化0期状态
                    h[t, m, l, 0:K] = S_initial[:]
                    h[t, m, l, K:2 * K] = E_initial[:]
                    h[t, m, l, 2 * K:3 * K] = A_initial[:]
                    h[t, m, l, 3 * K:4 * K] = U_initial[:]
                    h[t, m, l, 4 * K:5 * K] = b_before_time[:]
                    hh[t, m, l, 0:K] = Q_initial[:]
                    hh[t, m, l, K:2 * K] = R_initial[:]
                    hh[t, m, l, 2 * K:3 * K] = D_initial[:]

                S_tml = h[t, m, l, 0:K]  # 取出本期状态
                E_tml = h[t, m, l, K:2 * K]
                A_tml = h[t, m, l, 2 * K:3 * K]
                U_tml = h[t, m, l, 3 * K:4 * K]
                b_last_tml = h[t, m, l, 4 * K:5 * K]
                Q_tml = hh[t, m, l, 0:K]
                R_tml = hh[t, m, l, K:2 * K]
                D_tml = hh[t, m, l, 2 * K:3 * K]

                if 1 - 100 / (m + 1) > 0:
                    xi_hat_1 = 1 - 100 / m  # 计算轮盘赌随机数
                else:
                    xi_hat_1 = 0
                xi_hat_2 = xi_hat_1 + (1 - xi_hat_1) / 5
                xi_hat_3 = xi_hat_2 + (1 - xi_hat_1) / 5
                xi_hat_4 = xi_hat_3 + (1 - xi_hat_1) / 5
                xi_hat_5 = xi_hat_4 + (1 - xi_hat_1) / 5

                xi = random.uniform(0, 1)  # 开始轮盘赌，随机产生样本
                if xi < xi_hat_1:
                    print('还没写好ADP')
                elif xi < xi_hat_2:
                    b_tml, c_tml, value_myopic = myopic_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N
                                                              , sigma_hat, beta_e, beta_a, beta_u, eta
                                                              , b_hat[0: t + 1], lambda_b, C[t], lambda_c)
                    b_tml, c_tml = b_tml.reshape(K, 1), c_tml.reshape(K, 1)
                elif xi < xi_hat_3:
                    b_tml, c_tml, value_BH_Aver = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                            , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                            , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                            , eta, b_hat, C, 'Benchmark_Average', b_last_tml)
                elif xi < xi_hat_4:
                    b_tml, c_tml, value_BH_N = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                         , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                         , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                         , eta, b_hat, C, 'Benchmark_N', b_last_tml)
                elif xi < xi_hat_5:
                    b_tml, c_tml, value_BH_U = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                         , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                         , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                         , eta, b_hat, C, 'Benchmark_U', b_last_tml)
                else:
                    E_tml_last = E_initial_last
                    b_tml, c_tml, value_BH_U_n = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                           , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                           , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                           , eta, b_hat, C, 'Benchmark_U_new', b_last_tml, E_tml_last)

                # 状态转移，得到下一期状态
                S_nxt, E_nxt, A_nxt, Q_nxt, U_nxt, R_nxt, D_nxt \
                    = allocation_epidemic_function(K, 1, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml, N
                                                   , sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_q
                                                   , delta_u, gamma_a, gamma_q, gamma_u, p, q, b_tml, c_tml, eta)
                h[t + 1, m, l, 0:K] = S_nxt[:, 1]
                h[t + 1, m, l, K:2 * K] = E_nxt[:, 1]
                h[t + 1, m, l, 2 * K:3 * K] = A_nxt[:, 1]
                h[t + 1, m, l, 3 * K:4 * K] = U_nxt[:, 1]
                h[t + 1, m, l, 4 * K:5 * K] = b_tml[:, 0]
                hh[t + 1, m, l, 0:K] = Q_nxt[:, 1]
                hh[t + 1, m, l, K:2 * K] = R[:, 1]
                hh[t + 1, m, l, 2 * K:3 * K] = D[:, 1]

        e_s_time = time.time()
        print(f'正向采样，共用时{e_s_time - s_s_time}s')

        if m <= M - 2:
            continue

        s_a_time = time.time()
        train_x = np.zeros((T + 1, (m + 1) * L, 5 * K))  # 神经网络的训练集，每个时期t都是一行一个样本，符合sklearn（t=0时没有用）
        train_y = np.zeros((T + 1, (m + 1) * L))  # 神经网络的训练集，每个时期t都是一个元素一个样本标签，符合sklearn（t=0时没有用）
        for t in range(T, 0, -1):  # （2）逆向近似
            for m_ in range(m + 1):
                for l in range(L):
                    train_x[t, m_ * L + l, :] = h[t, m_, l, :]  # 放入样本
                    if t == T:  # 最后一期，求解当期规划问题即可 => 样本标签值
                        S_tml = h[t, m_, l, 0:K]
                        E_tml = h[t, m_, l, K:2 * K]
                        A_tml = h[t, m_, l, 2 * K:3 * K]
                        U_tml = h[t, m_, l, 3 * K:4 * K]
                        b_last_tml = h[t, m_, l, 4 * K:5 * K]
                        b_myopic, c_myopic, value_myopic = myopic_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N
                                                                        , sigma_hat, beta_e, beta_a, beta_u, eta
                                                                        , b_hat[0: t + 1], lambda_b, C[t], lambda_c)
                        train_y[t, m_ * L + l] = value_myopic
                    else:  # 求解ADP规划问题 => 样本标签值
                        S_tml = h[t, m_, l, 0:K]
                        E_tml = h[t, m_, l, K:2 * K]
                        A_tml = h[t, m_, l, 2 * K:3 * K]
                        U_tml = h[t, m_, l, 3 * K:4 * K]
                        b_last_tml = h[t, m_, l, 4 * K:5 * K]
                        b_ADP, c_ADP, value_ADP = adp_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N, sigma_hat
                                                            , beta_e, beta_a, beta_u, eta, b_hat[0: t + 1],lambda_b
                                                            , C[t], lambda_c, W, theta_1[t + 1, :, :], phi_1[t + 1, :]
                                                            , theta_2[t + 1, :], phi_2[t + 1], alpha, delta_a, delta_u
                                                            , gamma_a, gamma_u, p, q)
                        train_y[t, m_ * L + l] = value_ADP

            theta_1[t, :, :], theta_2[t, :], phi_1[t, :], phi_2[t], xx = adp_fnn(train_x[t, :, :], train_y[t, :], W)
            print(f'平均绝对值误差：{xx}')
        e_a_time = time.time()
        print(f'逆向近似，共用时{e_a_time - s_a_time}s')

        e_m_time = time.time()
        print(f'第{m + 1}次修正，结束，共用时{e_m_time - s_m_time}s')

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
        # print(f'{t}上一期：{b_last}')

        if t == T:
            b[:, t], c[:, t], value[t] = myopic_model(K, S_tem, E_tem, A_tem, U_tem, b_last, N
                                                      , sigma_hat, beta_e, beta_a, beta_u, eta
                                                      , b_hat[0: t + 1], lambda_b, C[t], lambda_c)
        else:
            b[:, t], c[:, t], value[t] = adp_model(K, S_tem, E_tem, A_tem, U_tem, b_last, N, sigma_hat
                                                   , beta_e, beta_a, beta_u, eta, b_hat[0: t + 1],lambda_b
                                                   , C[t], lambda_c, W, theta_1[t + 1, :, :], phi_1[t + 1, :]
                                                   , theta_2[t + 1, :], phi_2[t + 1], alpha, delta_a, delta_u
                                                   , gamma_a, gamma_u, p, q)
        # print(f'{t}本期决策：{b[:, t]}')
        S_nxt, E_nxt, A_nxt, Q_nxt\
            , U_nxt, R_nxt, D_nxt = allocation_epidemic_function(K, 1, S_tem, E_tem, A_tem, Q_tem
                                                                 , U_tem, R_tem, D_tem, N, sigma_hat
                                                                 , beta_e, beta_a, beta_u, alpha
                                                                 , delta_a, delta_q, delta_u
                                                                 , gamma_a, gamma_q, gamma_u
                                                                 , p, q, b[:, t:t + 1], c[:, t:t + 1], eta)
        if t != T:  # 更新下一期
            S[:, t + 1] = S_nxt[:, 1]  # S_nxt第一列是本期，第二列是下一期
            E[:, t + 1] = E_nxt[:, 1]
            A[:, t + 1] = A_nxt[:, 1]
            Q[:, t + 1] = Q_nxt[:, 1]
            U[:, t + 1] = U_nxt[:, 1]
            R[:, t + 1] = R_nxt[:, 1]
            D[:, t + 1] = D_nxt[:, 1]

    return b, c, value
