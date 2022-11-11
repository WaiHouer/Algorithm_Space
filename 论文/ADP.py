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
        , M, L, lambda_b, lambda_c, W
        , S_myopic, E_myopic, A_myopic, U_myopic, b_last_myopic
        , S_BH_Aver, E_BH_Aver, A_BH_Aver, U_BH_Aver, b_last_BH_Aver
        , S_BH_N, E_BH_N, A_BH_N, U_BH_N, b_last_BH_N
        , S_BH_U, E_BH_U, A_BH_U, U_BH_U, b_last_BH_U
        , S_BH_U_n, E_BH_U_n, A_BH_U_n, U_BH_U_n, b_last_BH_U_n
        , select_ratio, re_tag=None, b_before=None, B_add=0, norm_tag=None):
    # 初始化决策向量、目标函数值
    b, c = np.zeros((K, T + 1)), np.zeros((K, T + 1))
    value = np.zeros(T + 1)
    # T+1的目的：把起点放进来
    S, E, A, Q, U, R, D = np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)) \
        , np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))

    h = np.zeros((T + 1, M, L, 5 * K))  # 初始化状态矩阵
    hh = np.zeros((T + 1, M, L, 3 * K))  # 用于存放状态Q,R,D，只有在引用启发式时用得到
    if b_before is not None:
        b_before_time = b_before  # 为了短视ADP准备的
    else:
        b_before_time = np.zeros(K)  # 初始化时间跨度前一期的决策0
    theta_1 = np.zeros((T + 1, 5 * K, W))  # 隐藏层权值矩阵
    phi_1 = np.zeros((T + 1, W))  # 隐藏层偏置向量
    theta_2 = np.zeros((T + 1, W))  # 输出层权值向量
    phi_2 = np.zeros(T + 1)  # 输出层偏置值

    norm_up = np.zeros((T + 1, 5 * K))  # 均值归一分子
    norm_down = np.ones((T + 1, 5 * K))  # 均值归一分母

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

                if 1 - (M - select_ratio) / (m + 1) > 0:
                    xi_hat_1 = 1 - (M - select_ratio) / (m + 1)  # 计算轮盘赌随机数（ADP自身被选择概率不超过一定概率）
                else:
                    xi_hat_1 = 0
                xi_hat_2 = xi_hat_1 + (1 - xi_hat_1) / 5
                xi_hat_3 = xi_hat_2 + (1 - xi_hat_1) / 5
                xi_hat_4 = xi_hat_3 + (1 - xi_hat_1) / 5
                xi_hat_5 = xi_hat_4 + (1 - xi_hat_1) / 5

                xi = random.uniform(0, 1)  # 开始轮盘赌，随机产生样本
                if xi < xi_hat_1:
                    b_tml, c_tml, value_ADP = adp_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N, sigma_hat
                                                        , beta_e, beta_a, beta_u, eta, b_hat[0: t + 1], lambda_b
                                                        , C[t], lambda_c, W, theta_1[t + 1, :, :], phi_1[t + 1, :]
                                                        , theta_2[t + 1, :], phi_2[t + 1], alpha, delta_a, delta_u
                                                        , gamma_a, gamma_u, p, q
                                                        , norm_up[t + 1, :], norm_down[t + 1, :], B_add=B_add)
                    b_tml, c_tml = b_tml.reshape(K, 1), c_tml.reshape(K, 1)
                elif xi < xi_hat_2:
                    # print(t, b_last_tml)
                    b_tml, c_tml, value_myopic = myopic_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N
                                                              , sigma_hat, beta_e, beta_a, beta_u, eta
                                                              , b_hat[0: t + 1], lambda_b, C[t], lambda_c, B_add=B_add)
                    b_tml, c_tml = b_tml.reshape(K, 1), c_tml.reshape(K, 1)
                elif xi < xi_hat_3:
                    b_tml, c_tml, value_BH_Aver = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                            , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                            , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                            , eta, [b_hat[t]], C, 'Benchmark_Average', b_last_tml)
                elif xi < xi_hat_4:
                    b_tml, c_tml, value_BH_N = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                         , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                         , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                         , eta, [b_hat[t]], C, 'Benchmark_N', b_last_tml)
                elif xi < xi_hat_5:
                    b_tml, c_tml, value_BH_U = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                         , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                         , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                         , eta, [b_hat[t]], C, 'Benchmark_U', b_last_tml)
                else:
                    E_tml_last = E_initial_last
                    b_tml, c_tml, value_BH_U_n = benchmark(K, 0, S_tml, E_tml, A_tml, Q_tml, U_tml, R_tml, D_tml
                                                           , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a
                                                           , delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                                                           , eta, [b_hat[t]], C, 'Benchmark_U_new', b_last_tml, E_tml_last)

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

        if m + 1 < (M - select_ratio):  # ADP被纳入轮盘赌之前，不需要近似，只需采样即可
            continue

        s_a_time = time.time()
        train_x = np.zeros((T + 1, (m + 1) * L + 5, 5 * K))  # 神经网络的训练集，每个时期t都是一行一个样本，符合sklearn（t=0时没有用）
        train_y = np.zeros((T + 1, (m + 1) * L + 5))  # 神经网络的训练集，每个时期t都是一个元素一个样本标签，符合sklearn（t=0时没有用）
        for t in range(T, 0, -1):  # （2）逆向近似
            for m_ in range(m + 1):
                for l in range(L):
                    train_x[t, m_ * L + l, :] = h[t, m_, l, :]  # 放入样本
                    S_tml = h[t, m_, l, 0:K]
                    E_tml = h[t, m_, l, K:2 * K]
                    A_tml = h[t, m_, l, 2 * K:3 * K]
                    U_tml = h[t, m_, l, 3 * K:4 * K]
                    b_last_tml = h[t, m_, l, 4 * K:5 * K]
                    if t == T:  # 最后一期，求解当期规划问题即可 => 样本标签值
                        b_myopic, c_myopic, value_myopic = myopic_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N
                                                                        , sigma_hat, beta_e, beta_a, beta_u, eta
                                                                        , b_hat[0: t + 1], lambda_b, C[t], lambda_c
                                                                        , B_add=B_add)
                        train_y[t, m_ * L + l] = value_myopic
                    else:  # 求解ADP规划问题 => 样本标签值
                        b_ADP, c_ADP, value_ADP_total\
                            = adp_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N, sigma_hat
                                        , beta_e, beta_a, beta_u, eta, b_hat[0: t + 1],lambda_b
                                        , C[t], lambda_c, W, theta_1[t + 1, :, :], phi_1[t + 1, :]
                                        , theta_2[t + 1, :], phi_2[t + 1], alpha, delta_a, delta_u
                                        , gamma_a, gamma_u, p, q
                                        , norm_up[t + 1, :], norm_down[t + 1, :], re_tag=1, B_add=B_add)
                        train_y[t, m_ * L + l] = value_ADP_total

            # 将完整的其他算法链条，当做样本（1个myopic+4个启发式）
            h_myopic = np.hstack((S_myopic[:, t], E_myopic[:, t], A_myopic[:, t]
                                  , U_myopic[:, t], b_last_myopic[:, t - 1]))
            h_BH_Aver = np.hstack((S_BH_Aver[:, t], E_BH_Aver[:, t], A_BH_Aver[:, t]
                                   , U_BH_Aver[:, t], b_last_BH_Aver[:, t - 1]))
            h_BH_N = np.hstack((S_BH_N[:, t], E_BH_N[:, t], A_BH_N[:, t]
                                , U_BH_N[:, t], b_last_BH_N[:, t - 1]))
            h_BH_U = np.hstack((S_BH_U[:, t], E_BH_U[:, t], A_BH_U[:, t]
                                , U_BH_U[:, t], b_last_BH_U[:, t - 1]))
            h_BH_U_n = np.hstack((S_BH_U_n[:, t], E_BH_U_n[:, t], A_BH_U_n[:, t]
                                  , U_BH_U_n[:, t], b_last_BH_U_n[:, t - 1]))
            l_add = 0
            for h_tem in [h_myopic, h_BH_Aver, h_BH_N, h_BH_U, h_BH_U_n]:
                train_x[t, (m + 1) * L + l_add, :] = h_tem[:]
                S_tml = h_tem[0:K]
                E_tml = h_tem[K:2 * K]
                A_tml = h_tem[2 * K:3 * K]
                U_tml = h_tem[3 * K:4 * K]
                b_last_tml = h_tem[4 * K:5 * K]
                if t == T:
                    b_myopic, c_myopic, value_myopic = myopic_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N
                                                                    , sigma_hat, beta_e, beta_a, beta_u, eta
                                                                    , b_hat[0: t + 1], lambda_b, C[t], lambda_c
                                                                    , B_add=B_add)
                    train_y[t, (m + 1) * L + l_add] = value_myopic
                else:
                    b_ADP, c_ADP, value_ADP_total \
                        = adp_model(K, S_tml, E_tml, A_tml, U_tml, b_last_tml, N, sigma_hat
                                    , beta_e, beta_a, beta_u, eta, b_hat[0: t + 1], lambda_b
                                    , C[t], lambda_c, W, theta_1[t + 1, :, :], phi_1[t + 1, :]
                                    , theta_2[t + 1, :], phi_2[t + 1], alpha, delta_a, delta_u
                                    , gamma_a, gamma_u, p, q
                                    , norm_up[t + 1, :], norm_down[t + 1, :], re_tag=1, B_add=B_add)
                    train_y[t, (m + 1) * L + l_add] = value_ADP_total
                l_add += 1

            theta_1[t, :, :], theta_2[t, :], phi_1[t, :], phi_2[t], norm_up[t, :], norm_down[t, :] \
                = adp_fnn(train_x[t, :, :], train_y[t, :], W, norm_tag=norm_tag)
            print(f'- 第{t}期逆向近似，完成')
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
            b_last = b_before_time  # 上期病床
        else:
            b_last = b[:, t - 1]
        # print(f'{t}上一期：{b_last}')

        if t == T:
            b[:, t], c[:, t], value[t] = myopic_model(K, S_tem, E_tem, A_tem, U_tem, b_last, N
                                                      , sigma_hat, beta_e, beta_a, beta_u, eta
                                                      , b_hat[0: t + 1], lambda_b, C[t], lambda_c, B_add=B_add)
        else:
            b[:, t], c[:, t], value[t] = adp_model(K, S_tem, E_tem, A_tem, U_tem, b_last, N, sigma_hat
                                                   , beta_e, beta_a, beta_u, eta, b_hat[0: t + 1],lambda_b
                                                   , C[t], lambda_c, W, theta_1[t + 1, :, :], phi_1[t + 1, :]
                                                   , theta_2[t + 1, :], phi_2[t + 1], alpha, delta_a, delta_u
                                                   , gamma_a, gamma_u, p, q
                                                   , norm_up[t + 1, :], norm_down[t + 1, :], B_add=B_add)
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

    if re_tag:
        return b, c, value, S, E, A, Q, U, R, D
    else:
        return b, c, value
