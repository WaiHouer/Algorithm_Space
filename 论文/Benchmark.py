"""
基准启发式算法——平均分配、按人口、按确诊、按新增确诊
"""
import numpy as np
from Allocation_Epidemic_Function import allocation_epidemic_function
import math


def benchmark(K, T, S_initial, E_initial, A_initial, Q_initial, U_initial, R_initial, D_initial
              , N, sigma_hat, beta_e, beta_a, beta_u, alpha, delta_a, delta_q, delta_u
              , gamma_a, gamma_q, gamma_u, p, q, eta, b_hat, C, tag
              , b_last_tem, E_initial_last=None):
    # 初始化决策向量、目标函数值
    b, c = np.zeros((K, T + 1)), np.zeros((K, T + 1))
    value = np.zeros(T + 1)
    # T+1的目的：把起点放进来
    S, E, A, Q, U, R, D = np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)) \
        , np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))
    E_out = np.zeros(K)  # 把时间跨度以外的末期也记录下来，用于推导第T+1期value值

    for k in range(K):  # 起点初始化
        S[k][0] = S_initial[k]
        E[k][0] = E_initial[k]
        A[k][0] = A_initial[k]
        Q[k][0] = Q_initial[k]
        U[k][0] = U_initial[k]
        R[k][0] = R_initial[k]
        D[k][0] = D_initial[k]

    b_last = b_last_tem  # 上一期病床
    for t in range(T + 1):
        S_tem = S[:, t]  # 当期状态
        E_tem = E[:, t]
        A_tem = A[:, t]
        U_tem = U[:, t]
        Q_tem = Q[:, t]
        R_tem = R[:, t]
        D_tem = D[:, t]

        b_already, c_already = 0, 0
        if tag == 'Benchmark_Average':  # 启发式——平均分配
            for k in range(K):  # 平均分配（向下取整）
                b[k][t] = b_last[k] + math.floor(b_hat[t] / K)
                b_already += math.floor(b_hat[t] / K)
                c[k][t] = math.floor(C[t] / K)
                c_already += c[k][t]

            if b_hat[t] - b_already > 0:  # 剩余的随机分配
                index = np.random.randint(0, K - 1, size=int(b_hat[t] - b_already))
                for k in index:
                    b[k][t] += 1
            if C[t] - c_already > 0:
                index = np.random.randint(0, K - 1, size=int(C[t] - c_already))
                for k in index:
                    c[k][t] += 1
            b_last = b[:, t]

        elif tag == 'Benchmark_N':  # 启发式——按人口
            N_sum = 0
            for j in range(K):
                N_sum += N[j]
            for k in range(K):  # 按人口分配（向下取整）
                b[k][t] = b_last[k] + math.floor(b_hat[t] * N[k] / N_sum)
                b_already += math.floor(b_hat[t] * N[k] / N_sum)
                c[k][t] = math.floor(C[t] * N[k] / N_sum)
                c_already += c[k][t]

            if b_hat[t] - b_already > 0:  # 剩余的分配给N最多的
                index = list(N).index(max(N))
                b[index][t] += int(b_hat[t] - b_already)
            if C[t] - c_already > 0:
                index = list(N).index(max(N))
                c[index][t] += int(C[t] - c_already)
            b_last = b[:, t]

        elif tag == 'Benchmark_U':  # 启发式——按确诊
            U_sum, N_sum = 0, 0
            for j in range(K):
                U_sum += U[j][t]
                N_sum += N[j]
            for k in range(K):  # 按确诊分配（向下取整）
                b[k][t] = b_last[k] + math.floor(b_hat[t] * U[k][t] / U_sum)
                b_already += math.floor(b_hat[t] * U[k][t] / U_sum)
                c[k][t] = math.floor(C[t] * N[k] / N_sum)
                c_already += c[k][t]

            if b_hat[t] - b_already > 0:  # 剩余的分配给U（b）、N（c）最多的
                index = list(U[:, t]).index(max(U[:, t]))
                b[index][t] += int(b_hat[t] - b_already)
            if C[t] - c_already > 0:
                index = list(N).index(max(N))
                c[index][t] += int(C[t] - c_already)
            b_last = b[:, t]

        elif tag == 'Benchmark_U_new':  # 启发式——按新增确诊
            U_new_sum, N_sum = 0, 0
            U_new = np.zeros(K)  # t时期的新增确诊
            for j in range(K):
                if t == 0:
                    U_new[j] = (1 - p) * (1 - q) * alpha[j] * E_initial_last[j]
                else:
                    U_new[j] = A[j][t - 1] / N[j] * c[j][t - 1] \
                               + (1 - p) * (1 - q) * alpha[j] * E[j][t - 1]
                U_new_sum += U_new[j]
                N_sum += N[j]
            for k in range(K):  # 按新增确诊分配（向下取整）
                b[k][t] = b_last[k] + math.floor(b_hat[t] * U_new[k] / U_new_sum)
                b_already += math.floor(b_hat[t] * U_new[k] / U_new_sum)
                c[k][t] = math.floor(C[t] * N[k] / N_sum)
                c_already += c[k][t]

            if b_hat[t] - b_already > 0:  # 剩余的分配给U_new（b）、N（c）最多的
                index = list(U_new).index(max(U_new))
                b[index][t] += int(b_hat[t] - b_already)
            if C[t] - c_already > 0:
                index = list(N).index(max(N))
                c[index][t] += int(C[t] - c_already)
            b_last = b[:, t]

        # 注意：这里必须用b[:, t:t+1]，而不用b[:, t]，两者虽然在所包含元素上没有任何区别，但是👇
        # b[:, t:t+1]拥有双重下标[k][t]，b[:, t]只拥有一个下标[k]，前者符合allocation_epidemic_function函数对b的要求
        S_nxt, E_nxt, A_nxt, Q_nxt, U_nxt, R_nxt, D_nxt = allocation_epidemic_function(K, 1, S_tem, E_tem, A_tem, Q_tem
                                                                                       , U_tem, R_tem, D_tem, N, sigma_hat
                                                                                       , beta_e, beta_a, beta_u, alpha
                                                                                       , delta_a, delta_q, delta_u
                                                                                       , gamma_a, gamma_q, gamma_u
                                                                                       , p, q, b[:, t:t + 1]
                                                                                       , c[:, t:t + 1], eta)

        if t != T:  # 更新下一期
            S[:, t + 1] = S_nxt[:, 1]  # S_nxt第一列是本期，第二列是下一期
            E[:, t + 1] = E_nxt[:, 1]
            A[:, t + 1] = A_nxt[:, 1]
            Q[:, t + 1] = Q_nxt[:, 1]
            U[:, t + 1] = U_nxt[:, 1]
            R[:, t + 1] = R_nxt[:, 1]
            D[:, t + 1] = D_nxt[:, 1]
        else:
            E_out = E_nxt[:, 1]

    for t in range(T + 1):
        if t != T:  # 前面正常推导出value
            for k in range(K):
                value[t] += E[k][t+1] - E[k][t] + alpha[k] * E[k][t]
        else:
            for k in range(K):  # 最后一期需要用到时间跨度外的数据
                value[t] += E_out[k] - E[k][t] + alpha[k] * E[k][t]

    return b, c, value
