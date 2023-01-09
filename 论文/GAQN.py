"""
Python的非线性规划求解
"""
import random
import time

import numpy as np
from numba import njit
import math
from sympy import symbols, diff
from sympy.parsing.sympy_parser import parse_expr
from QN import QN
from Allocation_Epidemic_Function import allocation_epidemic_function
from duplicate_removal import duplicate_removal


random.seed(2333)
np.random.seed(2333)


class GAQN_Method:
    def __init__(self, K, T, S_initial, E_initial, A_initial, Q_initial, U_initial, R_initial, D_initial, N, sigma_hat
                 , beta_e, beta_a, beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u
                 , p, q, eta, b_hat, lambda_b, C, lambda_c, b_myopic, c_myopic, b_BH_Aver, c_BH_Aver, b_BH_N, c_BH_N
                 , b_BH_U, c_BH_U, b_BH_U_n, c_BH_U_n
                 , value_nature_list, value_nature_list_region):
        self.K, self.T = K, T
        self.S_initial, self.E_initial, self.A_initial, self.Q_initial = S_initial, E_initial, A_initial, Q_initial
        self.U_initial, self.R_initial, self.D_initial = U_initial, R_initial, D_initial
        self.N, self.sigma_hat = N, sigma_hat
        self.beta_e, self.beta_a, self.beta_u = beta_e, beta_a, beta_u
        self.alpha, self.delta_a, self.delta_q, self.delta_u = alpha, delta_a, delta_q, delta_u
        self.gamma_a, self.gamma_q, self.gamma_u = gamma_a, gamma_q, gamma_u
        self.p, self.q, self.eta = p, q, eta
        self.b_hat, self.lambda_b, self.C, self.lambda_c = b_hat, lambda_b, C, lambda_c

        self.b_myopic, self.c_myopic = b_myopic, c_myopic

        self.b_BH_Aver, self.c_BH_Aver = b_BH_Aver, c_BH_Aver  # 其他启发式解，仅用于生成初始种群
        self.b_BH_N, self.c_BH_N = b_BH_N, c_BH_N
        self.b_BH_U, self.c_BH_U = b_BH_U, c_BH_U
        self.b_BH_U_n, self.c_BH_U_n = b_BH_U_n, c_BH_U_n

        self.GA_T = 30  # 每GA_T期进行一次遗传算法（遗传周期）
        self.max_iter = 10000  # 最大迭代次数
        self.population = 160  # 种群数量
        self.value_nature_list = value_nature_list  # 用于存放自然状态的value，用于减去，计算适应度
        self.value_nature_list_region = value_nature_list_region  # 用于存放自然状态的value（更细，每个区域），用于精确变异
        self.SBX_eta = 0.3  # SBX交叉策略中的eta值，值越大，子代越接近父代
        self.cross_t_p = 0.8  # 发生交叉的概率（不一定每一期都进行交叉操作）
        self.mutation_eta = 0.3  # 多项式变异策略中的eta值
        self.mutation_p = 0.3  # 变异发生的概率（不一定每一期每一区域都进行变异操作）
        self.repair_p = 1  # 修复发生的概率（修复强度）
        self.repair_iteration = 100  # 每一次修复的最高迭代次数

        self.algorithm()

    def algorithm(self):

        # 首先进行编码，把 每期病床变量 转换成 每期新增的病床变量，体现累加的概念
        c_init_all = self.c_myopic
        b_init_all = coding(self.b_myopic)

        c_2_all = self.c_BH_Aver  # 用于初始化种群
        b_2_all = coding(self.b_BH_Aver)
        c_3_all = self.c_BH_N
        b_3_all = coding(self.b_BH_N)
        c_4_all = self.c_BH_U
        b_4_all = coding(self.b_BH_U)
        c_5_all = self.c_BH_U_n
        b_5_all = coding(self.b_BH_U_n)

        n_1, n_2, n_3, n_4 = 4, 10, 12, 60  # 初始解个数（per）、额外myopic个数、邻域解个数（per）、随机解个数
        # n_father_1, n_father_2 = 5, 25  # 父代种群精英保留个数（不重复）、父代种群轮盘赌保留数量
        # n_cross_1, n_cross_2 = 10, 50  # 交叉种群精英保留个数（不重复）、交叉种群轮盘赌保留数量
        # n_mutation_1, n_mutation_2 = 10, 50  # 变异种群精英保留个数（不重复）、变异种群轮盘赌保留数量

        n_father_1, n_father_2 = 5, 20  # 父代种群精英保留个数（不重复）、父代种群轮盘赌保留数量
        n_cross_1, n_cross_2 = 5, 40  # 交叉种群精英保留个数（不重复）、交叉种群轮盘赌保留数量
        n_mutation_1, n_mutation_2 = 5, 40  # 变异种群精英保留个数（不重复）、变异种群轮盘赌保留数量
        n_mutation_3, n_mutation_4 = 5, 40  # 精确变异（边际效应）种群精英保留个数（不重复）、变异种群轮盘赌保留数量

        # 每轮更新的参数
        T_tem = 0
        S_init, E_init, A_init, Q_init = self.S_initial, self.E_initial, self.A_initial, self.Q_initial
        U_init, R_init, D_init = self.U_initial, self.R_initial, self.D_initial  # 每轮初始状态
        final_sol = np.zeros((self.K, self.T + 1))  # 存放最终结果
        B_last = np.zeros(self.K)  # 记录之前的累积和，用于解码，把新增病床还原为累积病床

        while T_tem <= self.T:  # 每个遗传周期进行一次遗传算法，直到最后
            if T_tem + self.GA_T <= self.T:
                print(f'开始第{T_tem}期至第{T_tem + self.GA_T - 1}期')
                T_start, T_end = T_tem, T_tem + self.GA_T - 1
            elif T_tem == self.T:
                print(f'剩余第{T_tem}期，单期')
                T_start, T_end = T_tem, T_tem
            else:
                print(f'剩余第{T_tem}期至第{self.T}期，多期但不足遗传算法应用周期')
                T_start, T_end = T_tem, self.T

            b_init = b_init_all[:, T_start:T_end + 1]  # 取出相应时段的决策变量
            c_init = c_init_all[:, T_start:T_end + 1]
            value_nature = sum(self.value_nature_list[T_start:T_end + 1])  # 相应时段的自然状态value，用于计算适应度
            # 相应时段的自然状态value列表，用于精确变异
            value_nature_list_region = self.value_nature_list_region[:, T_start:T_end + 1]
            b_hat = self.b_hat[T_start:T_end + 1]  # 相应时段的病床新增可用量
            C = self.C[T_start:T_end + 1]  # 相应时段的核酸检测资源新增可用量

            b_2, c_2 = b_2_all[:, T_start:T_end + 1], c_2_all[:, T_start:T_end + 1]  # 用于初始化种群
            b_3, c_3 = b_3_all[:, T_start:T_end + 1], c_3_all[:, T_start:T_end + 1]
            b_4, c_4 = b_4_all[:, T_start:T_end + 1], c_4_all[:, T_start:T_end + 1]
            b_5, c_5 = b_5_all[:, T_start:T_end + 1], c_5_all[:, T_start:T_end + 1]

            '-------------------'
            # b_init[:, :], c_init[:, :] = b_2[:, :], c_2[:, :]  # 去掉myopic解，用于测试
            '-------------------'

            # 初始化种群
            population_all = [{'b': np.zeros((self.K, len(b_init[0, :]))), 'c': np.zeros((self.K, len(c_init[0, :]))),
                               'fitness': 0}
                              for i in range(self.population)]  # 所有个体的基因
            for po in range(0, n_1):  # 各种启发式各n_1个
                population_all[po]['b'][:, :], population_all[po]['c'][:, :] = b_init[:, :], c_init[:, :]
                population_all[po + n_1]['b'][:, :], population_all[po + n_1]['c'][:, :] = b_2[:, :], c_2[:, :]
                population_all[po + n_1 * 2]['b'][:, :], population_all[po + n_1 * 2]['c'][:, :] = b_3[:, :], c_3[:, :]
                population_all[po + n_1 * 3]['b'][:, :], population_all[po + n_1 * 3]['c'][:, :] = b_4[:, :], c_4[:, :]
                population_all[po + n_1 * 4]['b'][:, :], population_all[po + n_1 * 4]['c'][:, :] = b_5[:, :], c_5[:, :]
            for po in range(n_1 * 5, n_1 * 5 + n_2):  # 额外加n_2个myopic解
                population_all[po]['b'][:, :], population_all[po]['c'][:, :] = b_init[:, :], c_init[:, :]
            for po in range(n_1 * 5 + n_2, n_1 * 5 + n_2 + n_3):  # 各种邻域解各n_3个
                population_all[po]['b'][:, :], population_all[po]['c'][:, :] = b_init[:, :], c_init[:, :]
                r = random.randint(1, len(b_init[0, :]))  # 随机打乱r个时期
                for rr in range(r):
                    rt = random.randint(0, len(b_init[0, :]) - 1)  # 随机打乱哪一期
                    np.random.shuffle(population_all[po]['b'][:, rt])
                    np.random.shuffle(population_all[po]['c'][:, rt])

                population_all[po + n_3]['b'][:, :], population_all[po + n_3]['c'][:, :] = b_2[:, :], c_2[:, :]
                r = random.randint(1, len(b_init[0, :]))  # 随机打乱r个时期
                for rr in range(r):
                    rt = random.randint(0, len(b_init[0, :]) - 1)  # 随机打乱哪一期
                    np.random.shuffle(population_all[po + n_3]['b'][:, rt])
                    np.random.shuffle(population_all[po + n_3]['c'][:, rt])

                population_all[po + n_3 * 2]['b'][:, :], population_all[po + n_3 * 2]['c'][:, :] = b_3[:, :], c_3[:, :]
                r = random.randint(1, len(b_init[0, :]))  # 随机打乱r个时期
                for rr in range(r):
                    rt = random.randint(0, len(b_init[0, :]) - 1)  # 随机打乱哪一期
                    np.random.shuffle(population_all[po + n_3 * 2]['b'][:, rt])
                    np.random.shuffle(population_all[po + n_3 * 2]['c'][:, rt])

                population_all[po + n_3 * 3]['b'][:, :], population_all[po + n_3 * 3]['c'][:, :] = b_4[:, :], c_4[:, :]
                r = random.randint(1, len(b_init[0, :]))  # 随机打乱r个时期
                for rr in range(r):
                    rt = random.randint(0, len(b_init[0, :]) - 1)  # 随机打乱哪一期
                    np.random.shuffle(population_all[po + n_3 * 3]['b'][:, rt])
                    np.random.shuffle(population_all[po + n_3 * 3]['c'][:, rt])

                population_all[po + n_3 * 4]['b'][:, :], population_all[po + n_3 * 4]['c'][:, :] = b_5[:, :], c_5[:, :]
                r = random.randint(1, len(b_init[0, :]))  # 随机打乱r个时期
                for rr in range(r):
                    rt = random.randint(0, len(b_init[0, :]) - 1)  # 随机打乱哪一期
                    np.random.shuffle(population_all[po + n_3 * 4]['b'][:, rt])
                    np.random.shuffle(population_all[po + n_3 * 4]['c'][:, rt])
            for po in range(n_1 * 5 + n_2 + n_3 * 5, n_1 * 5 + n_2 + n_3 * 5 + n_4):  # 随机生成n_4个解
                for t in range(len(b_init[0, :])):
                    b_k_index, c_k_index = np.arange(0, self.K), np.arange(0, self.K)
                    np.random.shuffle(b_k_index), np.random.shuffle(c_k_index)  # 随机打乱的k下标
                    b_k_index, c_k_index = list(b_k_index), list(c_k_index)
                    while True:
                        if not b_k_index:
                            break
                        k = int(b_k_index.pop())  # 取出一个k下标
                        dif = b_hat[t] - sum(population_all[po]['b'][:, t])  # 当前距离上限还剩几个
                        if dif > b_hat[t] * self.lambda_b:
                            population_all[po]['b'][k][t] = random.randint(1, b_hat[t] * self.lambda_b)
                        elif 0 < dif <= b_hat[t] * self.lambda_b:
                            population_all[po]['b'][k][t] = dif
                        else:
                            break
                    while True:
                        if not c_k_index:
                            break
                        k = int(c_k_index.pop())  # 取出一个k下标
                        dif = C[t] - sum(population_all[po]['c'][:, t])  # 当前距离上限还剩几个
                        if dif > C[t] * self.lambda_c:
                            population_all[po]['c'][k][t] = random.randint(1, C[t] * self.lambda_c)
                        elif 0 < dif <= C[t] * self.lambda_c:
                            population_all[po]['c'][k][t] = dif
                        else:
                            break

            best_po = {'b': np.zeros((self.K, len(b_init[0, :]))), 'c': np.zeros((self.K, len(c_init[0, :]))),
                       'fitness': 0}  # 存储最好的个体
            for i in range(self.max_iter):
                print(f'第{i + 1}次遗传迭代：')
                # 计算适应度函数，同时存储本代最好的结果
                num_80000 = 0
                for po in range(self.population):
                    if i == 0:  # 如果是第一次进来，就先计算一下适应度
                        # 解码，注意：b_tem = population_all[po]['b'] 这种直接赋值方式会出错
                        b_tem, c_tem = np.zeros((self.K, len(b_init[0, :]))), np.zeros((self.K, len(b_init[0, :])))
                        b_tem[:, :] = population_all[po]['b'][:, :]
                        c_tem[:, :] = population_all[po]['c'][:, :]
                        b_tem = decoding(b_tem, B_last)
                        # 计算适应度函数
                        population_all[po]['fitness'] = \
                            fitness_calculate(self.K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, self.N
                                              , self.sigma_hat, self.beta_e, self.beta_a, self.beta_u, self.alpha
                                              , self.delta_a, self.delta_q, self.delta_u, self.gamma_a, self.gamma_q
                                              , self.gamma_u, self.p, self.q, b_tem, c_tem, self.eta, value_nature)

                    if population_all[po]['fitness'] > best_po['fitness']:  # 存储最好的
                        print('2322323   ', population_all[po]['fitness'], best_po['fitness'])
                        best_po['b'][:, :] = population_all[po]['b'][:, :]
                        best_po['c'][:, :] = population_all[po]['c'][:, :]
                        best_po['fitness'] = population_all[po]['fitness']
                    if population_all[po]['fitness'] > 80000:
                        num_80000 += 1
                    # print(po, population_all[po]['fitness'])
                print('当前最优适应度：', best_po['fitness'], f'大于80000的个体数 = {num_80000}')
                if i == 1000 or i == 2000 or i == 3000 or i == self.max_iter - 1:
                    print('当前最优解-b：', best_po['b'])
                    print('当前最优解-c：', best_po['c'])

                # 选择策略——选择个体进行接下来的交叉变异操作
                select_time = time.time()
                fitness_list = np.zeros(self.population)  # 创建一个适应度表，输入轮盘赌函数选择父代
                for po in range(self.population):
                    fitness_list[po] = population_all[po]['fitness']
                parents = selection_roulette(fitness_list, self.population)  # 轮盘赌选择策略
                print(f'--选择完毕，费时：{time.time() - select_time}s')

                # 交叉策略——生成新个体
                cross_time = time.time()
                population_new_1 = cross_SBX(self.K, len(b_init[0, :]), list(parents), population_all, self.SBX_eta
                                             , self.cross_t_p)
                print(f'--交叉完毕，费时：{time.time() - cross_time}s')

                # 变异策略——新个体有小概率变异
                mutation_time = time.time()
                population_new_2 = mutation_polynomial(self.K, len(b_init[0, :]), population_new_1, self.mutation_eta
                                                       , self.mutation_p)
                print(f'--变异完毕，费时：{time.time() - mutation_time}s')

                # 修复策略——对于不满足约束的个体进行修复（一次修复一个，输入：单个基因）
                repair_time = time.time()
                for po in range(len(population_new_1)):  # 修复交叉新种群
                    b_new, c_new = repair_gradient_based(self.K, len(b_init[0, :]), population_new_1[po]['b']
                                                         , population_new_1[po]['c'], self.repair_p, self.repair_iteration
                                                         , b_hat, C, self.lambda_b, self.lambda_c)
                    population_new_1[po]['b'][:, :] = b_new[:, :]
                    population_new_1[po]['c'][:, :] = c_new[:, :]
                for po in range(len(population_new_2)):  # 修复变异新种群
                    b_new, c_new = repair_gradient_based(self.K, len(b_init[0, :]), population_new_2[po]['b']
                                                         , population_new_2[po]['c'], self.repair_p, self.repair_iteration
                                                         , b_hat, C, self.lambda_b, self.lambda_c)
                    population_new_2[po]['b'][:, :] = b_new[:, :]
                    population_new_2[po]['c'][:, :] = c_new[:, :]
                print(f'--修复完毕，费时：{time.time() - repair_time}s')

                # 取整策略——四舍五入 + 随机找点补差（修复算法会把差值缩小到个位数，所以随机找一个补就行）
                round_time = time.time()
                for po in range(len(population_new_1)):  # 四舍五入（交叉）
                    population_new_1[po]['b'] = np.round(population_new_1[po]['b'])
                    population_new_1[po]['c'] = np.round(population_new_1[po]['c'])
                    for t in range(len(b_init[0, :])):
                        tem_sum_b = sum(population_new_1[po]['b'][:, t])
                        tem_sum_c = sum(population_new_1[po]['c'][:, t])
                        if tem_sum_b > b_hat[t]:  # 若b资源有超过的差值，随机找足够大的点补
                            dif = tem_sum_b - b_hat[t]
                            while True:
                                k = random.randint(0, self.K - 1)
                                if population_new_1[po]['b'][k][t] - dif >= 0:
                                    population_new_1[po]['b'][k][t] -= dif
                                    break
                        if tem_sum_c > C[t]:  # 若c资源有超过的差值，随机找足够大的点补
                            dif = tem_sum_c - C[t]
                            while True:
                                k = random.randint(0, self.K - 1)
                                if population_new_1[po]['c'][k][t] - dif >= 0:
                                    population_new_1[po]['c'][k][t] -= dif
                                    break

                for po in range(len(population_new_2)):  # 四舍五入（变异）
                    population_new_2[po]['b'] = np.round(population_new_2[po]['b'])
                    population_new_2[po]['c'] = np.round(population_new_2[po]['c'])
                    for t in range(len(b_init[0, :])):
                        tem_sum_b = sum(population_new_2[po]['b'][:, t])
                        tem_sum_c = sum(population_new_2[po]['c'][:, t])
                        if tem_sum_b > b_hat[t]:  # 若b资源有超过的差值，随机找足够大的点补
                            dif = tem_sum_b - b_hat[t]
                            while True:
                                k = random.randint(0, self.K - 1)
                                if population_new_2[po]['b'][k][t] - dif >= 0:
                                    population_new_2[po]['b'][k][t] -= dif
                                    break
                        if tem_sum_c > C[t]:  # 若c资源有超过的差值，随机找足够大的点补
                            dif = tem_sum_c - C[t]
                            while True:
                                k = random.randint(0, self.K - 1)
                                if population_new_2[po]['c'][k][t] - dif >= 0:
                                    population_new_2[po]['c'][k][t] -= dif
                                    break
                print(f'--取整完毕，费时：{time.time() - round_time}s')

                # 计算交叉和变异产生的个体的适应度
                for po in range(len(population_new_1)):
                    b_tem, c_tem = np.zeros((self.K, len(b_init[0, :]))), np.zeros((self.K, len(b_init[0, :])))
                    b_tem[:, :] = population_new_1[po]['b'][:, :]
                    c_tem[:, :] = population_new_1[po]['c'][:, :]
                    b_tem = decoding(b_tem, B_last)
                    # 计算适应度函数
                    population_new_1[po]['fitness'] = \
                        fitness_calculate(self.K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, self.N
                                          , self.sigma_hat, self.beta_e, self.beta_a, self.beta_u, self.alpha
                                          , self.delta_a, self.delta_q, self.delta_u, self.gamma_a, self.gamma_q
                                          , self.gamma_u, self.p, self.q, b_tem, c_tem, self.eta, value_nature)

                for po in range(len(population_new_2)):
                    b_tem, c_tem = np.zeros((self.K, len(b_init[0, :]))), np.zeros((self.K, len(b_init[0, :])))
                    b_tem[:, :] = population_new_2[po]['b'][:, :]
                    c_tem[:, :] = population_new_2[po]['c'][:, :]
                    b_tem = decoding(b_tem, B_last)
                    # 计算适应度函数
                    population_new_2[po]['fitness'] = \
                        fitness_calculate(self.K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, self.N
                                          , self.sigma_hat, self.beta_e, self.beta_a, self.beta_u, self.alpha
                                          , self.delta_a, self.delta_q, self.delta_u, self.gamma_a, self.gamma_q
                                          , self.gamma_u, self.p, self.q, b_tem, c_tem, self.eta, value_nature)

                # 精确变异策略——净增长边际效应
                mutation_time_2 = time.time()
                population_new_3 = [{'b': np.zeros((self.K, len(b_init[0, :])))
                                     , 'c': np.zeros((self.K, len(c_init[0, :])))
                                     , 'fitness': 0} for i in range(len(population_new_1))]
                for po in range(len(population_new_1)):  # 每次针对一个个体
                    r = random.random()
                    if r > self.mutation_p:  # 不满足变异概率
                        population_new_3[po]['b'][:, :] = population_new_1[po]['b'][:, :]
                        population_new_3[po]['c'][:, :] = population_new_1[po]['c'][:, :]
                        population_new_3[po]['fitness'] = population_new_1[po]['fitness']
                        continue
                    po_tem = \
                        mutation_fit(self.K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, self.N
                                     , self.sigma_hat, self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                     , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u, self.p
                                     , self.q, self.eta, value_nature_list_region, population_new_1[po], B_last
                                     , b_hat, C, self.lambda_b, self.lambda_c)
                    population_new_3[po]['b'][:, :] = po_tem['b'][:, :]
                    population_new_3[po]['c'][:, :] = po_tem['c'][:, :]
                    population_new_3[po]['fitness'] = po_tem['fitness']
                print(f'--精确变异（边际增长）完毕，费时：{time.time() - mutation_time_2}s')
                print(f'交叉产生新种群：{len(population_new_1)}'
                      f', 变异产生新种群：{len(population_new_2)}'
                      f', 精确变异（边际增长）产生新种群：{len(population_new_3)}')

                # 选择策略——从新个体中，选择相同数量作为新一代种群
                select_time_2 = time.time()
                population_all_new = []

                sort_ = sorted(population_all, key=lambda x:x['fitness'])  # 列表-字典，按键值排序（从小到大）
                duplicate_ = duplicate_removal(sort_, ['fitness'], model='key')  # 列表-字典，按键值去重
                for j in range(n_father_1):  # 父代精英保留不重复的 n_father_1 个
                    tem = duplicate_.pop()
                    population_all_new.append(tem)

                sort_ = sorted(population_new_1, key=lambda x:x['fitness'])  # 列表-字典，按键值排序（从小到大）
                duplicate_ = duplicate_removal(sort_, ['fitness'], model='key')  # 列表-字典，按键值去重
                for j in range(n_cross_1):  # 交叉精英保留不重复的 n_cross_1 个
                    tem = duplicate_.pop()
                    population_all_new.append(tem)

                sort_ = sorted(population_new_2, key=lambda x: x['fitness'])  # 列表-字典，按键值排序（从小到大）
                duplicate_ = duplicate_removal(sort_, ['fitness'], model='key')  # 列表-字典，按键值去重
                for j in range(n_mutation_1):  # 变异精英保留不重复的 n_mutation_1 个
                    tem = duplicate_.pop()
                    population_all_new.append(tem)

                sort_ = sorted(population_new_3, key=lambda x: x['fitness'])  # 列表-字典，按键值排序（从小到大）
                duplicate_ = duplicate_removal(sort_, ['fitness'], model='key')  # 列表-字典，按键值去重
                for j in range(n_mutation_3):  # 精确变异（边际增长）精英保留不重复的 n_mutation_3 个
                    tem = duplicate_.pop()
                    population_all_new.append(tem)

                fitness_list = np.zeros(self.population)
                for po in range(len(fitness_list)):  # 父代轮盘赌 n_father_2 个
                    fitness_list[po] = population_all[po]['fitness']
                select = selection_roulette(fitness_list, n_father_2, tag=1)
                for po in select:
                    tem = {'b': np.zeros((self.K, len(b_init[0, :]))), 'c': np.zeros((self.K, len(c_init[0, :]))),
                           'fitness': 0}
                    tem['b'][:, :] = population_all[int(po)]['b'][:, :]
                    tem['c'][:, :] = population_all[int(po)]['c'][:, :]
                    tem['fitness'] = population_all[int(po)]['fitness']
                    population_all_new.append(tem)

                fitness_list = np.zeros(int(self.population * 2))
                for po in range(len(fitness_list)):  # 交叉轮盘赌 n_cross_2 个
                    fitness_list[po] = population_new_1[po]['fitness']
                select = selection_roulette(fitness_list, n_cross_2, tag=1)
                for po in select:
                    tem = {'b': np.zeros((self.K, len(b_init[0, :]))), 'c': np.zeros((self.K, len(c_init[0, :]))),
                           'fitness': 0}
                    tem['b'][:, :] = population_new_1[int(po)]['b'][:, :]
                    tem['c'][:, :] = population_new_1[int(po)]['c'][:, :]
                    tem['fitness'] = population_new_1[int(po)]['fitness']
                    population_all_new.append(tem)

                fitness_list = np.zeros(int(self.population * 2))
                for po in range(len(fitness_list)):  # 变异轮盘赌 n_mutation_2 个
                    fitness_list[po] = population_new_2[po]['fitness']
                select = selection_roulette(fitness_list, n_mutation_2, tag=1)
                for po in select:
                    tem = {'b': np.zeros((self.K, len(b_init[0, :]))), 'c': np.zeros((self.K, len(c_init[0, :]))),
                           'fitness': 0}
                    tem['b'][:, :] = population_new_2[int(po)]['b'][:, :]
                    tem['c'][:, :] = population_new_2[int(po)]['c'][:, :]
                    tem['fitness'] = population_new_2[int(po)]['fitness']
                    population_all_new.append(tem)

                fitness_list = np.zeros(int(self.population * 2))
                for po in range(len(fitness_list)):  # 精确变异（边际增长）轮盘赌 n_mutation_4 个
                    fitness_list[po] = population_new_3[po]['fitness']
                select = selection_roulette(fitness_list, n_mutation_4, tag=1)
                for po in select:
                    tem = {'b': np.zeros((self.K, len(b_init[0, :]))), 'c': np.zeros((self.K, len(c_init[0, :]))),
                           'fitness': 0}
                    tem['b'][:, :] = population_new_3[int(po)]['b'][:, :]
                    tem['c'][:, :] = population_new_3[int(po)]['c'][:, :]
                    tem['fitness'] = population_new_3[int(po)]['fitness']
                    population_all_new.append(tem)

                print(f'--最终选择完毕，费时：{time.time() - select_time_2}s')

                # 更新下一代
                for po in range(self.population):
                    population_all[po]['b'][:, :] = population_all_new[po]['b'][:, :]
                    population_all[po]['c'][:, :] = population_all_new[po]['c'][:, :]
                    population_all[po]['fitness'] = population_all_new[po]['fitness']

            # S, E, A, Q, U, R, D, B_last更新


            T_tem += self.GA_T
            break

    def algorithm_1(self):
        T_tem = 0
        S, E, A, Q = self.S_initial, self.E_initial, self.A_initial, self.Q_initial  # 每两期更新一次
        U, R, D = self.U_initial, self.R_initial, self.D_initial
        b_last = np.zeros(self.K)
        while T_tem <= 3:
            if T_tem + 1 <= 3:
                print(f'开始第{T_tem}期至第{T_tem + 1}期')
                b_init = self.b_myopic[:, T_tem:T_tem + 2]  # 取出这两期的初始解
                c_init = self.c_myopic[:, T_tem:T_tem + 2]
                B = [sum(self.b_hat[0:T_tem + 1]), sum(self.b_hat[0:T_tem + 2])]  # 这两期的病床总和上限
                s_time = time.time()
                QN(self.K, S, E, A, U, self.N, self.sigma_hat, self.beta_e, self.beta_a, self.beta_u, self.alpha
                   , self.delta_a, self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                   , self.p, self.q, self.eta, self.b_hat[T_tem: T_tem + 2], self.lambda_b
                   , self.C[T_tem: T_tem + 2], self.lambda_c, b_init, c_init, B, b_last)
                print(f'拟牛顿{time.time() - s_time}s')



            else:
                print(f'剩余第{T_tem}期，直接采用myopic')



            T_tem += 2


@ njit()
def coding(b):  # 编码
    K, T = len(b), len(b[0, :])
    b_code = np.zeros((K, T))
    for k in range(K):
        for t in range(T):
            if t == 0:
                b_code[k][t] = b[k][t]
            else:
                b_code[k][t] = b[k][t] - b[k][t - 1]
    return b_code


@ njit()
def decoding(b, B_last):  # 解码
    K, T = len(b), len(b[0, :])
    b_decode = b
    for t in range(T):  # 解码（新增病床 -> 累计病床）
        for k in range(K):
            if t == 0:
                b_decode[k][t] += B_last[k]
            else:
                b_decode[k][t] += b_decode[k][t - 1]
    return b_decode


def fitness_calculate(K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, N, sigma_hat, beta_e, beta_a
                      , beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q, b, c, eta
                      , value_nature, tag=None):
    T = len(b[0, :])
    fitness = 0
    S, E, A, Q, U, R, D = np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1)) \
        , np.zeros((K, T + 1)), np.zeros((K, T + 1)), np.zeros((K, T + 1))

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
                                     , b[:, t:t+1], c[:, t:t+1], eta)

        if t != T:  # 更新下一期
            S[:, t + 1] = S_nxt[:, 1]  # S_nxt第一列是本期，第二列是下一期
            E[:, t + 1] = E_nxt[:, 1]
            A[:, t + 1] = A_nxt[:, 1]
            Q[:, t + 1] = Q_nxt[:, 1]
            U[:, t + 1] = U_nxt[:, 1]
            R[:, t + 1] = R_nxt[:, 1]
            D[:, t + 1] = D_nxt[:, 1]

    for t in range(T):
        for k in range(K):
            fitness += S[k][t] - S[k][t + 1]
    fitness = value_nature - fitness

    if tag:
        return S, E, A, Q, U, R, D

    return fitness


def selection_roulette(fitness_list, population, tag=None):  # 选择策略——轮盘赌
    p = np.zeros(len(fitness_list))  # 概率表
    fitness_sum = sum(fitness_list)  # 适应度总和
    for po in range(len(fitness_list)):
        p[po] = fitness_list[po] / fitness_sum

    roulette_p = np.zeros(len(fitness_list))  # 轮盘赌概率表（即：累积分布）
    for po in range(len(fitness_list)):
        if po == 0:
            roulette_p[po] = p[po]
        else:
            roulette_p[po] = roulette_p[po - 1] + p[po]

    if tag is None:  # tag=None，就是代表第一次选择
        parents = np.zeros(population * 2)  # 存放父代下标
        for i in range(population * 2):
            r = random.random()
            for po in range(len(fitness_list)):
                if r <= roulette_p[po]:
                    index = po
                    break
            parents[i] = index

        return parents

    if tag:  # tag存在，就是代表第二次选择
        select = np.zeros(population)
        for i in range(population):
            r = random.random()
            for po in range(len(fitness_list)):
                if r <= roulette_p[po]:
                    index = po
                    break
            select[i] = index

        return select



def cross_SBX(K, T, parents, population_all, SBX_eta, cross_t_p):  # 交叉：模拟二进制交叉
    population_new = []
    while parents:
        parent_1 = population_all[int(parents.pop())]
        parent_2 = population_all[int(parents.pop())]
        son_1_b, son_2_b = np.zeros((K, T)), np.zeros((K, T))
        son_1_c, son_2_c = np.zeros((K, T)), np.zeros((K, T))
        r = random.random()
        if r > cross_t_p:  # 先判断交叉概率，决定这两个父代是否需要交叉，不需要的话则continue
            son_1_b[:, :], son_2_b[:, :] = parent_1['b'][:, :], parent_2['b'][:, :]  # 不交叉就直接遗传下来
            son_1_c[:, :], son_2_c[:, :] = parent_1['c'][:, :], parent_2['c'][:, :]
            population_new.append({'b': son_1_b, 'c': son_1_c, 'fitness': 0})
            population_new.append({'b': son_2_b, 'c': son_2_c, 'fitness': 0})
            continue
        cross_num = random.randint(1, T)  # 随机产生，需要交叉的时期t数量
        cross_t = list(np.arange(0, T))
        cross_t = random.sample(cross_t, cross_num)  # 随机选取规定数量的不重复的时期
        not_cross_t = [t for t in range(T) if t not in cross_t]  # 剩下的，不需要交叉的时期

        for t in not_cross_t:  # 不需要交叉的，直接遗传下来
            son_1_b[:, t], son_2_b[:, t] = parent_1['b'][:, t], parent_2['b'][:, t]
            son_1_c[:, t], son_2_c[:, t] = parent_1['c'][:, t], parent_2['c'][:, t]

        for t in cross_t:  # 需要交叉的，进行模拟二进制交叉

            r_b = 0  # 首先对于b资源是否交叉（0代表肯定交叉）
            if r_b <= cross_t_p:
                for k in range(K):
                    rand = random.random()
                    if rand <= 0.5:
                        beta = (rand * 2) ** (1 / (1 + SBX_eta))
                    else:
                        beta = (1 / (2 - rand * 2)) ** (1 / (1 + SBX_eta))

                    son_1_b[k][t] = 0.5 * ((1 + beta) * parent_1['b'][k][t] + (1 - beta) * parent_2['b'][k][t])
                    son_1_b[k][t] = max(son_1_b[k][t], 0)  # 我自己加的，防止产生小于0的
                    son_2_b[k][t] = 0.5 * ((1 - beta) * parent_1['b'][k][t] + (1 + beta) * parent_2['b'][k][t])
                    son_2_b[k][t] = max(son_2_b[k][t], 0)
            else:
                son_1_b[:, t], son_2_b[:, t] = parent_1['b'][:, t], parent_2['b'][:, t]  # 不交叉就直接遗传下来

            r_c = 0  # 其次对于c资源是否交叉（0代表肯定交叉）
            if r_c <= cross_t_p:
                for k in range(K):
                    rand = random.random()
                    if rand <= 0.5:
                        beta = (rand * 2) ** (1 / (1 + SBX_eta))
                    else:
                        beta = (1 / (2 - rand * 2)) ** (1 / (1 + SBX_eta))

                    son_1_c[k][t] = 0.5 * ((1 + beta) * parent_1['c'][k][t] + (1 - beta) * parent_2['c'][k][t])
                    son_1_c[k][t] = max(son_1_c[k][t], 0)  # 我自己加的，防止产生小于0的
                    son_2_c[k][t] = 0.5 * ((1 - beta) * parent_1['c'][k][t] + (1 + beta) * parent_2['c'][k][t])
                    son_2_c[k][t] = max(son_2_c[k][t], 0)
            else:
                son_1_c[:, t], son_2_c[:, t] = parent_1['c'][:, t], parent_2['c'][:, t]  # 不交叉就直接遗传下来

        population_new.append({'b': son_1_b, 'c': son_1_c, 'fitness': 0})
        population_new.append({'b': son_2_b, 'c': son_2_c, 'fitness': 0})

    return population_new


def mutation_polynomial(K, T, population_new, mutation_eta, mutation_p):
    population_new_2 = [{'b': np.zeros((K, T)), 'c': np.zeros((K, T)), 'fitness': 0}
                        for i in range(len(population_new))]
    for po in range(len(population_new_2)):  # 初始化赋值
        population_new_2[po]['b'][:, :] = population_new[po]['b'][:, :]
        population_new_2[po]['c'][:, :] = population_new[po]['c'][:, :]

    b_up_list, b_low_list = np.zeros((K, T)), np.ones((K, T)) * 99999999  # 存放各变量最大最小值
    c_up_list, c_low_list = np.zeros((K, T)), np.ones((K, T)) * 99999999
    for t in range(T):
        for k in range(K):
            for po in range(len(population_new_2)):
                if population_new_2[po]['b'][k][t] > b_up_list[k][t]:
                    b_up_list[k][t] = population_new_2[po]['b'][k][t]
                if population_new_2[po]['b'][k][t] < b_low_list[k][t]:
                    b_low_list[k][t] = population_new_2[po]['b'][k][t]
                if population_new_2[po]['c'][k][t] > c_up_list[k][t]:
                    c_up_list[k][t] = population_new_2[po]['c'][k][t]
                if population_new_2[po]['c'][k][t] < c_low_list[k][t]:
                    c_low_list[k][t] = population_new_2[po]['c'][k][t]

    for po in range(len(population_new_2)):
        r = random.random()
        if r > mutation_p:  # 不需要变异的，直接遗传下来
            continue

        mutation_t = random.randint(0, T - 1)  # 随机产生一期，对它进行变异
        mutation_k_b = random.randint(0, K - 1)  # 随机产生一区域，对它进行b资源变异
        mutation_k_c = random.randint(0, K - 1)  # 随机产生一区域，对它进行c资源变异

        v = population_new_2[po]['b'][mutation_k_b][mutation_t]  # 首先，对b进行变异
        u, l = b_up_list[mutation_k_b][mutation_t], b_low_list[mutation_k_b][mutation_t]
        if u == l:
            delta_1, delta_2 = 0, 0
        else:
            delta_1 = (v - l) / (u - l)
            delta_2 = (u - v) / (u - l)
        rand = random.random()
        if rand <= 0.5:
            delta = (
                    2 * rand + (1 - 2 * rand) * ((1 - delta_1) ** (mutation_eta + 1))
                    ) ** (1 / (mutation_eta + 1)) - 1
        else:
            delta = 1 - (
                    2 * (1 - rand) + 2 * (rand - 0.5) * ((1 - delta_2) ** (mutation_eta + 1))
                    ) ** (1 / (mutation_eta + 1))
        v = v + delta * (u - l)
        v = min(u, max(v, l))  # 有的代码会加这一条
        v = max(v, 0)  # 同样，防止小于0
        population_new_2[po]['b'][mutation_k_b][mutation_t] = v
        '--------------------'
        v = population_new_2[po]['c'][mutation_k_c][mutation_t]  # 其次，对c进行变异
        u, l = c_up_list[mutation_k_c][mutation_t], c_low_list[mutation_k_c][mutation_t]
        if u == l:
            delta_1, delta_2 = 0, 0
        else:
            delta_1 = (v - l) / (u - l)
            delta_2 = (u - v) / (u - l)
        rand = random.random()
        if rand <= 0.5:
            delta = (
                    2 * rand + (1 - 2 * rand) * ((1 - delta_1) ** (mutation_eta + 1))
                    ) ** (1 / (mutation_eta + 1)) - 1
        else:
            delta = 1 - (
                    2 * (1 - rand) + 2 * (rand - 0.5) * ((1 - delta_2) ** (mutation_eta + 1))
                    ) ** (1 / (mutation_eta + 1))
        v = v + delta * (u - l)
        v = min(u, max(v, l))  # 有的代码会加这一条
        v = max(v, 0)  # 同样，防止小于0
        population_new_2[po]['c'][mutation_k_c][mutation_t] = v

    return population_new_2


def repair_gradient_based(K, T, b, c, repair_p, repair_iteration, b_hat, C, lambda_b, lambda_c):
    b_new, c_new = np.zeros((K, T)), np.zeros((K, T))
    b_new[:, :], c_new[:, :] = b[:, :], c[:, :]
    x_num = 2 * K * T  # 变量个数：顺序是K个b_t0、K个b_t1、...、K个c_t0、K个c_t1、...

    for i in range(repair_iteration):
        delta_V, gradient_V = [], []  # 约束矩阵、约束的一阶导数矩阵

        for t in range(T):  # 第一组约束：当期新增病床量上限约束
            tem_delta = min(0, b_hat[t] - sum(b_new[:, t]))
            if tem_delta < -0.1:
                delta_V.append(tem_delta)
                tem_gradient = [0 for i in range(x_num)]
                tem_gradient[K * t:K * t + K] = [1 for k in range(K)]
                gradient_V.append(tem_gradient)
        for t in range(T):  # 第二组约束：当期新增核酸资源上限约束
            tem_delta = min(0, C[t] - sum(c_new[:, t]))
            if tem_delta < -0.1:
                delta_V.append(tem_delta)
                tem_gradient = [0 for i in range(x_num)]
                tem_gradient[K * T + K * t:K * T + K * t + K] = [1 for k in range(K)]
                gradient_V.append(tem_gradient)
        for t in range(T):  # 第三组约束：病床资源不超过部署率
            right_num = lambda_b * b_hat[t]
            for k in range(K):
                tem_delta = min(0, right_num - b_new[k][t])
                if tem_delta < -0.1:
                    delta_V.append(tem_delta)
                    tem_gradient = [0 for i in range(x_num)]
                    tem_gradient[K * t + k] = 1
                    gradient_V.append(tem_gradient)
        for t in range(T):  # 第四组约束：核酸资源不超过部署率
            right_num = lambda_c * C[t]
            for k in range(K):
                tem_delta = min(0, right_num - c_new[k][t])
                if tem_delta < -0.1:
                    delta_V.append(tem_delta)
                    tem_gradient = [0 for i in range(x_num)]
                    tem_gradient[K * T + K * t + k] = 1
                    gradient_V.append(tem_gradient)
        for t in range(T):  # 第五组约束：b变量不小于0
            for k in range(K):
                tem_delta = max(0, 0 - b_new[k][t])
                if tem_delta > 0.1:
                    delta_V.append(tem_delta)
                    tem_gradient = [0 for i in range(x_num)]
                    tem_gradient[K * t + k] = 1
                    gradient_V.append(tem_gradient)
        for t in range(T):  # 第六组约束：c变量不小于0
            for k in range(K):
                tem_delta = max(0, 0 - c_new[k][t])
                if tem_delta > 0.1:
                    delta_V.append(tem_delta)
                    tem_gradient = [0 for i in range(x_num)]
                    tem_gradient[K * T + K * t + k] = 1
                    gradient_V.append(tem_gradient)

        if not delta_V:  # 如果首次进来，发现没有违反约束，那么退出
            break
        if i == 0:  # 如果首次进来，修复概率没有触发，那么退出
            r = random.random()
            if r > repair_p:
                break

        MP_V = np.linalg.pinv(gradient_V)  # 一阶导数矩阵的广义逆矩阵
        delta_V = np.array(delta_V).reshape((len(delta_V), 1))  # 转换为一列，用于矩阵乘法
        delta_x = np.dot(MP_V, delta_V)  # 决策变量的更新量

        if np.linalg.norm(delta_x) < 0.01:  # 范数太小，那么退出
            break

        for t in range(T):  # 更新决策变量
            for k in range(K):
                b_new[k][t] = b_new[k][t] + delta_x[K * t + k]
                c_new[k][t] = c_new[k][t] + delta_x[K * T + K * t + k]

    return b_new, c_new


def mutation_fit(K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, N, sigma_hat, beta_e, beta_a
                 , beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q, eta, value_nature
                 , population, B_last, b_hat, C, lambda_b, lambda_c):
    # 精确变异算子：根据净增长的边际效应确定（每次进行一个个体）
    T = len(population['b'][0, :])  # 决策时期数（全总共29期）
    population_new = {'b': np.zeros((K, T)), 'c': np.zeros((K, T)), 'fitness': 0}

    b_tem, c_tem = np.zeros((K, T)), np.zeros((K, T))
    b_tem[:, :], c_tem[:, :] = population['b'][:, :], population['c'][:, :]
    b_tem = decoding(b_tem, B_last)  # 记得先解码，才能计算
    value_tem = np.sum(value_nature[:, :])
    S, E, A, Q, U, R, D = \
        fitness_calculate(K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, N, sigma_hat, beta_e, beta_a
                          , beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                          , b_tem, c_tem, eta, value_tem, tag=1)
    increase_sequence = np.zeros((K, T))  # 增长列表
    for t in range(T):
        for k in range(K):
            increase_sequence[k][t] = S[k][t] - S[k][t + 1]  # 净增长

    rt = random.randint(0, T - 1)  # 随机选取一期
    b_tem_, c_tem_ = np.zeros((K, T)), np.zeros((K, T))
    b_tem_[:, :], c_tem_[:, :] = b_tem[:, :], c_tem[:, :]
    marginal_total = np.zeros(K)
    for kt in range(K):  # 每个区域计算一遍边际效应
        b_tem_[kt][rt] += 10
        # c_tem_[kt][rt] += 1000
        S_, E_, A_, Q_, U_, R_, D_ = \
            fitness_calculate(K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, N, sigma_hat, beta_e, beta_a
                              , beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                              , b_tem_, c_tem_, eta, value_tem, tag=1)
        increase_sequence_ = np.zeros((K, T))  # 增长列表
        for t in range(T):
            for k in range(K):
                increase_sequence_[k][t] = S_[k][t] - S_[k][t + 1]  # 净增长
        marginal_total[kt] = np.sum(increase_sequence[kt, rt:T]) - np.sum(increase_sequence_[kt, rt:T])  # 边际效应 = 差值
        b_tem_[kt][rt] -= 10
        # c_tem_[kt][rt] -= 1000  # 记得减回去
    marginal_order = np.argsort(marginal_total)  # 按照边际效应大小，从小到大排序，对应的下标

    b_tem = coding(b_tem)  # 记得编码，变成更新量
    if np.sum(b_tem[:, rt]) < b_hat[rt]:  # 如果资源未满，则补充满
        gap_total = b_hat[rt] - np.sum(b_tem[:, rt])
        # print(f'资源未充分利用，补充：{gap_total}')
        for k in range(K - 1, -1, -1):  # 从边际效应最高的往前
            # print(f':{marginal_order[k]}')
            gap_tem = lambda_b * b_hat[rt] - b_tem[marginal_order[k]][rt]
            if gap_total <= gap_tem:
                # print(f'增加{gap_total}')
                b_tem[marginal_order[k]][rt] += gap_total
                gap_total = 0
            else:
                # print(f'增加{gap_tem}')
                b_tem[marginal_order[k]][rt] += gap_tem
                gap_total -= gap_tem
            if gap_total == 0:
                break

    num_b = 10  # 每次补充/交换10个b资源
    left, right = 0, K - 1  # 左对应边际效应最低的，右对应边际效应最高的
    while left < right:
        left_index, right_index = marginal_order[left], marginal_order[right]
        if b_tem[left_index][rt] == 0:  # 没有足够资源交换，往后走一个
            left += 1
            continue
        if b_tem[right_index][rt] == b_hat[rt] * lambda_b:  # 达到上限不能接收资源，往前走一个
            right -= 1
            continue

        num = min(b_tem[left_index][rt], b_hat[rt] * lambda_b - b_tem[right_index][rt])  # 两区域之间最多可以交换的数量
        if num >= num_b:
            b_tem[left_index][rt] -= num_b
            b_tem[right_index][rt] += num_b
            num_b = 0
        else:
            b_tem[left_index][rt] -= num
            b_tem[right_index][rt] += num
            num_b -= num

        if num_b == 0:
            break

    # 预留：c资源同样

    # 创建并返回新的变异population
    population_new['b'][:, :], population_new['c'][:, :] = b_tem[:, :], c_tem[:, :]
    b_tem = decoding(b_tem, B_last)
    fitness = \
        fitness_calculate(K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, N, sigma_hat, beta_e, beta_a
                          , beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q
                          , b_tem, c_tem, eta, value_tem)
    population_new['fitness'] = fitness
    return population_new
