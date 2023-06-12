"""
变邻域搜索算法
"""
import random
import time
import numpy as np
from GA_Basic_Function import coding, decoding, fitness_calculate
from Allocation_Epidemic_Function import allocation_epidemic_function
import math
import matplotlib.pyplot as plt


# random.seed(2333)
# np.random.seed(2333)


class Variable_Neighborhood_Search:
    def __init__(self, K, S_init, E_init, A_init, Q_init, U_init, R_init, D_init, N, sigma_hat, beta_e, beta_a
                 , beta_u, alpha, delta_a, delta_q, delta_u, gamma_a, gamma_q, gamma_u, p, q, eta, value_nature
                 , b_bar_init, c_init, B_last, b_hat, C, lambda_b, lambda_c, b_init):
        self.ttt = time.time()
        self.K = K
        self.S_init, self.E_init, self.A_init, self.Q_init = S_init, E_init, A_init, Q_init
        self.U_init, self.R_init, self.D_init = U_init, R_init, D_init
        self.N = N
        self.sigma_hat = sigma_hat
        self.beta_e, self.beta_a, self.beta_u = beta_e, beta_a, beta_u
        self.alpha, self.delta_a, self.delta_q, self.delta_u = alpha, delta_a, delta_q, delta_u
        self.gamma_a, self.gamma_q, self.gamma_u = gamma_a, gamma_q, gamma_u
        self.p, self.q, self.eta, self.value_nature = p, q, eta, value_nature
        self.b_bar_init, self.c_init = b_bar_init, c_init  # 分配量（更新量）
        self.B_last, self.b_hat, self.C = B_last, b_hat, C
        self.lambda_b, self.lambda_c = lambda_b, lambda_c
        self.b_init = b_init  # 使用量（全量）

        self.iter = 50  # 变邻域搜索迭代次数
        self.b_final, self.c_final, self.fitness_final = np.zeros_like(self.b_init), np.zeros_like(self.c_init), 0
        self.b_bar_final = np.zeros_like(self.b_bar_init)

        self.sol_num = 2  # 并行计算数量
        self.num_b = 5  # b资源，单时期内边际效用计算量（1%）
        self.num_c = 5000  # c资源，单时期内边际效用计算量（1%）
        self.t_num_b = 5  # b资源，双时期转移边际量（1%）
        self.t_num_c = 5000  # c资源，双时期转移边际量（1%）
        self.local_iter = 5  # 局部搜索的迭代上限
        self.shake_iter = 5  # 超过若干次的变邻域搜索迭代，无改进的话，就进行扰动
        self.difference = False  # 是否添加插值运算解

        self.fitness_list = [0 for i in range(self.iter + 1)]  # 记录每次迭代fit（首位元素=myopic）
        self.time_list = [0 for i in range(self.iter + 1)]  # 记录每次迭代时间（首位元素=0）

        self.algorithm()
        # self.picture_iter()  # 画出迭代拐点图

    def algorithm(self):
        b_bar_best, c_best = np.zeros_like(self.b_bar_init), np.zeros_like(self.c_init)  # 临时最优解
        b_best = np.zeros_like(self.b_init)
        fitness_best = 0

        b_bar, c = np.zeros_like(self.b_bar_init), np.zeros_like(self.c_init)  # 分配量
        b = np.zeros_like(self.b_init)  # 使用量
        b_bar[:, :], b[:, :], c[:, :] = self.b_bar_init[:, :], self.b_init[:, :], self.c_init[:, :]  # 初始解

        fitness = 0
        value_tem = np.sum(self.value_nature[:, :])  # 自然状态下群体，用于计算适应度时相减

        shake_iter = 0  # 记录当前多少次迭代无改进
        for i in range(self.iter):  # 变邻域搜索迭代
            print(f'第{i + 1}次迭代')
            fitness = \
                fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init, self.U_init, self.R_init
                                  , self.D_init, self.N, self.sigma_hat, self.beta_e, self.beta_a, self.beta_u
                                  , self.alpha, self.delta_a, self.delta_q, self.delta_u, self.gamma_a, self.gamma_q
                                  , self.gamma_u, self.p, self.q, b, c, self.eta, value_tem)
            if i == 0:  # 记录初始值
                self.fitness_list[0] = fitness
                self.time_list[0] = 0

            neighborhood_out = 1  # 外层邻域结构，初始化
            sol_list_out = [{'b_bar_out': np.zeros_like(b_bar), 'b_out': np.zeros_like(b)
                            , 'c_out': np.zeros_like(c), 'f_out': 0}
                            for i in range(self.sol_num)]
            while neighborhood_out <= 7:  # 所有邻域结构（外循环—变换邻域）
                if neighborhood_out == 1:
                    # b_bar_out, b_out, c_out = self.neighborhood_1(b_bar, b, c, value_tem)
                    for s in range(self.sol_num):
                        b_bar_out, b_out, c_out = self.neighborhood_1(b_bar, b, c, value_tem)
                        f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                  self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                  , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                  , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                                  , self.p,self.q, b_out, c_out, self.eta, value_tem)
                        sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                        sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                        sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                        sol_list_out[s]['f_out'] = f_out
                elif neighborhood_out == 2:
                    # b_bar_out, b_out, c_out = self.neighborhood_2(b_bar, b, c)
                    for s in range(self.sol_num):
                        b_bar_out, b_out, c_out = self.neighborhood_2(b_bar, b, c)
                        f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                  self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                  , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                  , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                                  , self.p,self.q, b_out, c_out, self.eta, value_tem)
                        sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                        sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                        sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                        sol_list_out[s]['f_out'] = f_out
                elif neighborhood_out == 3:
                    # b_bar_out, b_out, c_out = self.neighborhood_3(b_bar, b, c, value_tem)
                    for s in range(self.sol_num):
                        b_bar_out, b_out, c_out = self.neighborhood_3(b_bar, b, c, value_tem)
                        f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                  self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                  , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                  , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                                  , self.p,self.q, b_out, c_out, self.eta, value_tem)
                        sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                        sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                        sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                        sol_list_out[s]['f_out'] = f_out
                elif neighborhood_out == 4:
                    # b_bar_out, b_out, c_out = self.neighborhood_4(b_bar, b, c)
                    for s in range(self.sol_num):
                        b_bar_out, b_out, c_out = self.neighborhood_4(b_bar, b, c)
                        f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                  self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                  , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                  , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                                  , self.p,self.q, b_out, c_out, self.eta, value_tem)
                        sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                        sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                        sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                        sol_list_out[s]['f_out'] = f_out
                elif neighborhood_out == 5:
                    # b_bar_out, b_out, c_out = self.neighborhood_5(b_bar, b, c)
                    for s in range(self.sol_num):
                        b_bar_out, b_out, c_out = self.neighborhood_5(b_bar, b, c)
                        f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                  self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                  , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                  , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                                  , self.p,self.q, b_out, c_out, self.eta, value_tem)
                        sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                        sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                        sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                        sol_list_out[s]['f_out'] = f_out
                elif neighborhood_out == 6:
                    # b_bar_out, b_out, c_out = self.neighborhood_6(b_bar, b, c)
                    for s in range(self.sol_num):
                        b_bar_out, b_out, c_out = self.neighborhood_6(b_bar, b, c)
                        f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                  self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                  , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                  , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                                  , self.p,self.q, b_out, c_out, self.eta, value_tem)
                        sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                        sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                        sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                        sol_list_out[s]['f_out'] = f_out
                elif neighborhood_out == 7:
                    # b_bar_out, b_out, c_out = self.neighborhood_7(b_bar, b, c)
                    for s in range(self.sol_num):
                        b_bar_out, b_out, c_out = self.neighborhood_7(b_bar, b, c)
                        f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                  self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                  , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                  , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                                  , self.p, self.q, b_out, c_out, self.eta, value_tem)
                        sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                        sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                        sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                        sol_list_out[s]['f_out'] = f_out

                if self.sol_num > 1 and self.difference is True:  # 如果并行多个解，则进行插值运算，额外产生解
                    tem = {'b_bar_out': np.zeros_like(b_bar), 'b_out': np.zeros_like(b)
                           , 'c_out': np.zeros_like(c), 'f_out': 0}
                    index = [s for s in range(self.sol_num)]
                    select = random.sample(index, 2)  # 随机选两个解出来，进行差值运算

                    tem['b_bar_out'][:, :], tem['b_out'][:, :], tem['c_out'][:, :] \
                        = self.difference_cal(sol_list_out[select[0]]['b_bar_out'], sol_list_out[select[1]]['b_bar_out']
                                              , sol_list_out[select[0]]['b_out'], sol_list_out[select[1]]['b_out']
                                              , sol_list_out[select[0]]['c_out'], sol_list_out[select[1]]['c_out'])
                    f_out = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                              self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                              , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                              , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                                              , self.p,self.q, tem['b_out'], tem['c_out'], self.eta, value_tem)
                    tem['f_out'] = f_out
                    print('sss', f_out)
                    if f_out < 110000:
                        print('origin')
                        print(1, sol_list_out[select[0]]['b_bar_out'])
                        print(2, sol_list_out[select[1]]['b_bar_out'])
                        print(1, sol_list_out[select[0]]['b_out'])
                        print(2, sol_list_out[select[0]]['b_out'])
                        print(tem['b_bar_out'])
                        print(tem['b_out'])
                    sol_list_out.append(tem)

                for s in range(len(sol_list_out)):  # 对于每一个外循环解，进行内循环+局部搜索
                    neighborhood_in = 1  # 内层邻域结构，初始化
                    b_bar_out = sol_list_out[s]['b_bar_out']
                    b_out = sol_list_out[s]['b_out']
                    c_out = sol_list_out[s]['c_out']
                    f_out = sol_list_out[s]['f_out']
                    while neighborhood_in <= 7:  # 所有邻域结构（内循环—局部搜索）
                        if neighborhood_in == 1:
                            b_bar_in, b_in, c_in = self.neighborhood_1(b_bar_out, b_out, c_out, value_tem)
                        elif neighborhood_in == 2:
                            b_bar_in, b_in, c_in = self.neighborhood_2(b_bar_out, b_out, c_out)
                        elif neighborhood_in == 3:
                            b_bar_in, b_in, c_in = self.neighborhood_3(b_bar_out, b_out, c_out, value_tem)
                        elif neighborhood_in == 4:
                            b_bar_in, b_in, c_in = self.neighborhood_4(b_bar_out, b_out, c_out)
                        elif neighborhood_in == 5:
                            b_bar_in, b_in, c_in = self.neighborhood_5(b_bar_out, b_out, c_out)
                        elif neighborhood_in == 6:
                            b_bar_in, b_in, c_in = self.neighborhood_6(b_bar_out, b_out, c_out)
                        elif neighborhood_in == 7:
                            b_bar_in, b_in, c_in = self.neighborhood_7(b_bar_out, b_out, c_out)

                        f_in = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                 self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                 , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                 , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q
                                                 , self.gamma_u, self.p, self.q, b_in, c_in, self.eta, value_tem)

                        # 局部搜索，寻找内循环解的局部最优解（寻找方式：同样是循环邻域）
                        local_iter = 0  # 若local_iter次无提升，则退出
                        while True:
                            if neighborhood_in == 1:
                                b_bar_in_tem, b_in_tem, c_in_tem = self.neighborhood_1(b_bar_in, b_in, c_in, value_tem)
                            elif neighborhood_in == 2:
                                b_bar_in_tem, b_in_tem, c_in_tem = self.neighborhood_2(b_bar_in, b_in, c_in)
                            elif neighborhood_in == 3:
                                b_bar_in_tem, b_in_tem, c_in_tem = self.neighborhood_3(b_bar_in, b_in, c_in, value_tem)
                            elif neighborhood_in == 4:
                                b_bar_in_tem, b_in_tem, c_in_tem = self.neighborhood_4(b_bar_in, b_in, c_in)
                            elif neighborhood_in == 5:
                                b_bar_in_tem, b_in_tem, c_in_tem = self.neighborhood_5(b_bar_in, b_in, c_in)
                            elif neighborhood_in == 6:
                                b_bar_in_tem, b_in_tem, c_in_tem = self.neighborhood_6(b_bar_in, b_in, c_in)
                            elif neighborhood_in == 7:
                                b_bar_in_tem, b_in_tem, c_in_tem = self.neighborhood_7(b_bar_in, b_in, c_in)
                            f_in_tem = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                                     self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                                     , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                                     , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q
                                                     , self.gamma_u, self.p, self.q, b_in_tem, c_in_tem, self.eta, value_tem)
                            if f_in_tem > f_in:
                                local_iter = 0  # 重置
                                b_bar_in[:, :] = b_bar_in_tem[:, :]
                                b_in[:, :] = b_in_tem[:, :]
                                c_in[:, :] = c_in_tem[:, :]
                                f_in = f_in_tem
                                continue
                            else:
                                local_iter += 1
                            if local_iter > self.local_iter:
                                break

                        if f_in > f_out:
                            b_bar_out[:, :] = b_bar_in[:, :]
                            b_out[:, :] = b_in[:, :]
                            c_out[:, :] = c_in[:, :]
                            f_out = f_in
                            neighborhood_in = 1
                        else:
                            neighborhood_in += 1

                    sol_list_out[s]['b_bar_out'][:, :] = b_bar_out[:, :]
                    sol_list_out[s]['b_out'][:, :] = b_out[:, :]
                    sol_list_out[s]['c_out'][:, :] = c_out[:, :]
                    sol_list_out[s]['f_out'] = f_out

                update_tag = 0
                for s in range(self.sol_num):  # 如果搜索到了更优解，则更新
                    if sol_list_out[s]['f_out'] > fitness:
                        update_tag += 1
                        b_bar[:, :] = sol_list_out[s]['b_bar_out'][:, :]
                        b[:, :] = sol_list_out[s]['b_out'][:, :]
                        c[:, :] = sol_list_out[s]['c_out'][:, :]
                        fitness = sol_list_out[s]['f_out']
                if update_tag > 0:
                    print(fitness)
                    neighborhood_out = 1
                else:
                    neighborhood_out += 1  # 如果没有，则跳转下一邻域

            if fitness > self.fitness_final:  # 记录【最终最优解】
                self.b_bar_final[:, :] = b_bar[:, :]
                self.b_final[:, :] = b[:, :]
                self.c_final[:, :] = c[:, :]
                self.fitness_final = fitness

            self.fitness_list[i + 1] = self.fitness_final  # 记录迭代结果
            self.time_list[i + 1] = time.time() - self.ttt
            print(self.fitness_final)
            # print(self.b_bar_final)

            tag = 0
            if fitness > fitness_best:  # 判断【临时最优解】
                tag += 1
                b_bar_best[:, :] = b_bar[:, :]
                b_best[:, :] = b[:, :]
                c_best[:, :] = c[:, :]
                fitness_best = fitness
            if tag > 0:
                shake_iter = 0
            else:
                shake_iter += 1  # 如果没改进，则计数1次

            if shake_iter > self.shake_iter:  # 如果很多次没有改进，则进行扰动
                neighborhood = random.choice([2, 4, 5, 6, 7])  # 从随机性的邻域中，选一个进行扰动
                if neighborhood == 2:
                    b_bar_tem, b_tem, c_tem = self.neighborhood_2(b_bar, b, c)
                elif neighborhood == 4:
                    b_bar_tem, b_tem, c_tem = self.neighborhood_4(b_bar, b, c)
                elif neighborhood == 5:
                    b_bar_tem, b_tem, c_tem = self.neighborhood_5(b_bar, b, c)
                elif neighborhood == 6:
                    b_bar_tem, b_tem, c_tem = self.neighborhood_6(b_bar, b, c)
                elif neighborhood == 7:
                    b_bar_tem, b_tem, c_tem = self.neighborhood_7(b_bar, b, c)
                f_tem = fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init,
                                         self.U_init, self.R_init, self.D_init, self.N, self.sigma_hat
                                         , self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a
                                         , self.delta_q, self.delta_u, self.gamma_a, self.gamma_q
                                         , self.gamma_u, self.p, self.q, b_tem, c_tem, self.eta, value_tem)
                b_bar[:, :], b[:, :], c[:, :], fitness = b_bar_tem[:, :], b_tem[:, :], c_tem[:, :], f_tem
                # 同时，【临时最优解】同步更新成为扰动后的结果
                b_bar_best[:, :], b_best[:, :], c_best[:, :] = b_bar_tem[:, :], b_tem[:, :], c_tem[:, :]
                fitness_best = fitness

        # print(self.b_bar_final)
        # print(self.b_final)
        # print(self.c_final)

    def neighborhood_1(self, b_bar, b, c, value_tem):  # b资源-边际效应邻域
        T = len(b[0, :])
        b_bar_tem, b_tem, c_tem = np.zeros_like(b_bar), np.zeros_like(b), np.zeros_like(c)
        b_bar_tem[:, :], b_tem[:, :], c_tem[:, :] = b_bar[:, :], b[:, :], c[:, :]  # 赋值临时变量

        S, E, A, Q, U, R, D = \
            fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init, self.U_init, self.R_init
                              , self.D_init, self.N, self.sigma_hat, self.beta_e, self.beta_a, self.beta_u, self.alpha
                              , self.delta_a, self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                              , self.p, self.q, b_tem, c_tem, self.eta, value_tem, tag=1)
        increase_seq = np.zeros((self.K, T))  # 增长明细列表
        for t in range(T):
            for k in range(self.K):
                increase_seq[k][t] = S[k][t] - S[k][t + 1]

        rt = random.randint(0, T - 1)  # 随机选取一期

        margin = np.zeros(self.K)  # 每个区域的边际效应
        for kt in range(self.K):  # 针对每个区域

            b_bar_tem = decoding(b_bar_tem, self.B_last)  # 成为累积分配量
            if b_bar_tem[kt][rt] == b_tem[kt][rt]:  # 如果累积分配量=使用量，代表该区域能用尽资源（简单来看，不考虑约束问题）
                b_bar_tem[kt][rt] += self.num_b  # 分配量+一个边际量
                b_tem[kt][rt] += self.num_b  # 那么，使用量也会+一个边际量

            S_, E_, A_, Q_, U_, R_, D_ = \
                fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init, self.U_init,
                                  self.R_init, self.D_init, self.N, self.sigma_hat, self.beta_e, self.beta_a
                                  , self.beta_u, self.alpha, self.delta_a, self.delta_q, self.delta_u, self.gamma_a
                                  , self.gamma_q, self.gamma_u, self.p, self.q, b_tem, c_tem, self.eta, value_tem
                                  , tag=1)
            increase_seq_ = np.zeros((self.K, T))  # 增长明细列表（增加资源后）
            for t in range(T):
                for k in range(self.K):
                    increase_seq_[k][t] = S_[k][t] - S_[k][t + 1]

            margin[kt] = np.sum(increase_seq[kt, rt:T]) - np.sum(increase_seq_[kt, rt:T])

            b_bar_tem = coding(b_bar_tem)  # 还原成为更新量
            b_bar_tem[:, :], b_tem[:, :] = b_bar[:, :], b[:, :]  # 恢复初始解
        margin_order = np.argsort(margin)  # 按照边际效应大小，从小到大排序

        # 针对分配量，进行资源转移操作
        num_b = random.randint(0, self.b_hat[rt])  # 每次转移，不超过num_b个b资源
        left, right = 0, self.K - 1  # 左对应边际效应最低的，右对应边际效应最高的
        while left < right:
            left_index, right_index = margin_order[left], margin_order[right]
            if b_bar_tem[left_index][rt] == 0:  # 没有足够资源交换，往后走一个
                left += 1
                continue
            if b_bar_tem[right_index][rt] == self.b_hat[rt] * self.lambda_b:  # 达到上限不能接收资源，往前走一个
                right -= 1
                continue
            # 两区域之间最多可以交换的数量
            num = min(b_bar_tem[left_index][rt], self.b_hat[rt] * self.lambda_b - b_bar_tem[right_index][rt])
            if num >= num_b:
                b_bar_tem[left_index][rt] -= num_b
                b_bar_tem[right_index][rt] += num_b
                num_b = 0
            else:
                b_bar_tem[left_index][rt] -= num
                b_bar_tem[right_index][rt] += num
                num_b -= num
            if num_b == 0:
                break

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b, decoding(b_bar_new, self.B_last), c_tem, rt, T)
        return b_bar_tem, b_tem, c_tem

    def neighborhood_2(self, b_bar, b, c):  # b资源-随机交换邻域
        T = len(b[0, :])
        b_bar_tem, b_tem, c_tem = np.zeros_like(b_bar), np.zeros_like(b), np.zeros_like(c)
        b_bar_tem[:, :], b_tem[:, :], c_tem[:, :] = b_bar[:, :], b[:, :], c[:, :]  # 赋值临时变量

        rt = random.randint(0, T - 1)  # 随机选取一期

        k_sub_list, k_add_list = [], []  # 能够转出和转入的区域集合
        for k in range(self.K):
            if b_bar_tem[k][rt] > 0:
                k_sub_list.append(k)
            if b_bar_tem[k][rt] < self.b_hat[rt] * self.lambda_b:
                k_add_list.append(k)

        if k_sub_list == [] or k_add_list == []:  # 如果没有区域可以用，那么直接终止
            return b_bar_tem, b_tem, c_tem

        if len(k_sub_list) == 1:  # 如果只有一个区域可以进行
            k_sub = k_sub_list[0]
            while True:
                k_add = random.choice(k_add_list)  # 随机选一期不同的，转入资源
                if k_add != k_sub:
                    break
        elif len(k_add_list) == 1:
            k_add = k_add_list[0]
            while True:
                k_sub = random.choice(k_sub_list)  # 随机选一期不同的，转入资源
                if k_add != k_sub:
                    break
        else:  # 如果有多种选择，就随机选择
            k_sub = random.choice(k_sub_list)  # 随机选一期转出资源
            while True:
                k_add = random.choice(k_add_list)  # 随机选一期不同的，转入资源
                if k_add != k_sub:
                    break

        # 针对分配量，进行资源转移操作
        num_b = random.randint(0, self.b_hat[rt])  # 每次转移，不超过num_b个b资源
        # 两区域之间最多可以交换的数量
        num = min(b_bar_tem[k_sub][rt], self.b_hat[rt] * self.lambda_b - b_bar_tem[k_add][rt])
        num = random.randint(0, num)
        b_bar_tem[k_sub][rt] -= num
        b_bar_tem[k_add][rt] += num

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b, decoding(b_bar_new, self.B_last), c_tem, rt, T)
        return b_bar_tem, b_tem, c_tem

    def neighborhood_3(self, b_bar, b, c, value_tem):  # c资源-边际效应邻域
        T = len(b[0, :])
        b_bar_tem, b_tem, c_tem = np.zeros_like(b_bar), np.zeros_like(b), np.zeros_like(c)
        b_bar_tem[:, :], b_tem[:, :], c_tem[:, :] = b_bar[:, :], b[:, :], c[:, :]  # 赋值临时变量

        S, E, A, Q, U, R, D = \
            fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init, self.U_init, self.R_init
                              , self.D_init, self.N, self.sigma_hat, self.beta_e, self.beta_a, self.beta_u, self.alpha
                              , self.delta_a, self.delta_q, self.delta_u, self.gamma_a, self.gamma_q, self.gamma_u
                              , self.p, self.q, b_tem, c_tem, self.eta, value_tem, tag=1)
        increase_seq = np.zeros((self.K, T))  # 增长明细列表
        for t in range(T):
            for k in range(self.K):
                increase_seq[k][t] = S[k][t] - S[k][t + 1]

        rt = random.randint(0, T - 1)  # 随机选取一期

        margin = np.zeros(self.K)  # 每个区域的边际效应
        for kt in range(self.K):  # 针对每个区域
            c_tem[kt][rt] += self.num_c

            S_, E_, A_, Q_, U_, R_, D_ = \
                fitness_calculate(self.K, self.S_init, self.E_init, self.A_init, self.Q_init, self.U_init,
                                  self.R_init, self.D_init, self.N, self.sigma_hat, self.beta_e, self.beta_a
                                  , self.beta_u, self.alpha, self.delta_a, self.delta_q, self.delta_u, self.gamma_a
                                  , self.gamma_q, self.gamma_u, self.p, self.q, b_tem, c_tem, self.eta, value_tem
                                  , tag=1)
            increase_seq_ = np.zeros((self.K, T))  # 增长明细列表（增加资源后）
            for t in range(T):
                for k in range(self.K):
                    increase_seq_[k][t] = S_[k][t] - S_[k][t + 1]

            margin[kt] = np.sum(increase_seq[kt, rt:T]) - np.sum(increase_seq_[kt, rt:T])

            c_tem[:, :] = c[:, :]  # 恢复初始解
        margin_order = np.argsort(margin)  # 按照边际效应大小，从小到大排序

        # 针对分配量，进行资源转移操作
        num_c = random.randint(0, self.C[rt])  # 每次转移，不超过num_c个c资源
        left, right = 0, self.K - 1  # 左对应边际效应最低的，右对应边际效应最高的
        while left < right:
            left_index, right_index = margin_order[left], margin_order[right]
            if c_tem[left_index][rt] == 0:  # 没有足够资源交换，往后走一个
                left += 1
                continue
            if c_tem[right_index][rt] == self.C[rt] * self.lambda_c:  # 达到上限不能接收资源，往前走一个
                right -= 1
                continue
            # 两区域之间最多可以交换的数量
            num = min(c_tem[left_index][rt], self.C[rt] * self.lambda_c - c_tem[right_index][rt])
            if num >= num_c:
                c_tem[left_index][rt] -= num_c
                c_tem[right_index][rt] += num_c
                num_c = 0
            else:
                c_tem[left_index][rt] -= num
                c_tem[right_index][rt] += num
                num_c -= num
            if num_c == 0:
                break

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b, decoding(b_bar_new, self.B_last), c_tem, rt, T)
        return b_bar_tem, b_tem, c_tem

    def neighborhood_4(self, b_bar, b, c):  # c资源-随机交换邻域
        T = len(b[0, :])
        b_bar_tem, b_tem, c_tem = np.zeros_like(b_bar), np.zeros_like(b), np.zeros_like(c)
        b_bar_tem[:, :], b_tem[:, :], c_tem[:, :] = b_bar[:, :], b[:, :], c[:, :]  # 赋值临时变量

        rt = random.randint(0, T - 1)  # 随机选取一期

        k_sub_list, k_add_list = [], []  # 能够转出和转入的区域集合
        for k in range(self.K):
            if c_tem[k][rt] > 0:
                k_sub_list.append(k)
            if c_tem[k][rt] < self.C[rt] * self.lambda_c:
                k_add_list.append(k)

        if k_sub_list == [] or k_add_list == []:  # 如果没有区域可以用，那么直接终止
            return b_bar_tem, b_tem, c_tem

        if len(k_sub_list) == 1:  # 如果只有一个区域可以进行
            k_sub = k_sub_list[0]
            while True:
                k_add = random.choice(k_add_list)  # 随机选一期不同的，转入资源
                if k_add != k_sub:
                    break
        elif len(k_add_list) == 1:
            k_add = k_add_list[0]
            while True:
                k_sub = random.choice(k_sub_list)  # 随机选一期不同的，转入资源
                if k_add != k_sub:
                    break
        else:  # 如果有多种选择，就随机选择
            k_sub = random.choice(k_sub_list)  # 随机选一期转出资源
            while True:
                k_add = random.choice(k_add_list)  # 随机选一期不同的，转入资源
                if k_add != k_sub:
                    break

        # 针对分配量，进行资源转移操作
        num_c = random.randint(0, self.C[rt])  # 每次转移，不超过num_c个c资源
        # 两区域之间最多可以交换的数量
        num = min(c_tem[k_sub][rt], self.C[rt] * self.lambda_c - c_tem[k_add][rt])
        num = random.randint(0, num)
        c_tem[k_sub][rt] -= num
        c_tem[k_add][rt] += num
        # if num >= num_c:
        #     c_tem[k_sub][rt] -= num_c
        #     c_tem[k_add][rt] += num_c
        # else:
        #     c_tem[k_sub][rt] -= num
        #     c_tem[k_add][rt] += num

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b, decoding(b_bar_new, self.B_last), c_tem, rt, T)
        return b_bar_tem, b_tem, c_tem

    def neighborhood_5(self, b_bar, b, c):
        b_bar_tem, b_tem, c_tem = np.zeros_like(b_bar), np.zeros_like(b), np.zeros_like(c)
        b_bar_tem[:, :], b_tem[:, :], c_tem[:, :] = b_bar[:, :], b[:, :], c[:, :]

        T = len(b_bar[0, :])
        t_1 = random.randint(0, T - 1)  # 两时期，整体交换病床资源
        t_2 = random.randint(0, T - 1)
        tem = np.zeros((self.K, 1))

        tem[:, 0] = b_bar_tem[:, t_1]
        b_bar_tem[:, t_1] = b_bar_tem[:, t_2]
        b_bar_tem[:, t_2] = tem[:, 0]

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b, decoding(b_bar_new, self.B_last), c_tem, min(t_1, t_2), T)

        return b_bar_tem, b_tem, c_tem

    def neighborhood_6(self, b_bar, b, c):
        b_bar_tem, b_tem, c_tem = np.zeros_like(b_bar), np.zeros_like(b), np.zeros_like(c)
        b_bar_tem[:, :], b_tem[:, :], c_tem[:, :] = b_bar[:, :], b[:, :], c[:, :]

        T = len(b_bar[0, :])
        t_1 = random.randint(0, T - 1)  # 两时期，整体交换核酸检测资源
        t_2 = random.randint(0, T - 1)
        tem = np.zeros((self.K, 1))

        tem[:, 0] = c_tem[:, t_1]
        c_tem[:, t_1] = c_tem[:, t_2]
        c_tem[:, t_2] = tem[:, 0]

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b, decoding(b_bar_new, self.B_last), c_tem, min(t_1, t_2), T)

        return b_bar_tem, b_tem, c_tem

    def neighborhood_7(self, b_bar, b, c):
        b_bar_tem, b_tem, c_tem = np.zeros_like(b_bar), np.zeros_like(b), np.zeros_like(c)
        b_bar_tem[:, :], b_tem[:, :], c_tem[:, :] = b_bar[:, :], b[:, :], c[:, :]

        T = len(b_bar[0, :])
        select = random.sample([t for t in range(T)], 2)  # 随机选出两个不同时期
        t_1 = min(select[0], select[1])  # t_1转出资源，只能选择靠前的时期
        t_2 = max(select[0], select[1])  # t_2转入资源，只能选择靠后的时期（因为资源只能累积向后转移）

        t_1_list, t_2_list = [], []  # 能够转出和转入的区域集合
        for k in range(self.K):
            if b_bar_tem[k][t_1] > 0:
                t_1_list.append(k)
            if b_bar_tem[k][t_2] < self.b_hat[t_2] * self.lambda_b:
                t_2_list.append(k)

        if t_1_list == [] or t_2_list == []:  # 如果没有区域可以用，那么直接终止
            return b_bar_tem, b_tem, c_tem

        if len(t_1_list) == 1:  # 如果只有一个区域可以进行
            t_1_k = t_1_list[0]
            while True:
                t_2_k = random.choice(t_2_list)  # 随机选一期不同的，转入资源
                if t_1_k != t_2_k:
                    break
        elif len(t_2_list) == 1:
            t_2_k = t_2_list[0]
            while True:
                t_1_k = random.choice(t_1_list)  # 随机选一期不同的，转入资源
                if t_1_k != t_2_k:
                    break
        else:  # 如果有多种选择，就随机选择
            t_1_k = random.choice(t_1_list)  # 随机选一期转出资源
            while True:
                t_2_k = random.choice(t_2_list)  # 随机选一期不同的，转入资源
                if t_1_k != t_2_k:
                    break

        # 针对分配量，进行资源转移操作
        num_c = self.t_num_b  # 每次转移，不超过t_num_b个b资源
        # 两区域之间最多可以交换的数量
        num = min(b_bar_tem[t_1_k][t_1], self.b_hat[t_2] * self.lambda_b - b_bar_tem[t_2_k][t_2])
        if num >= num_c:
            b_bar_tem[t_1_k][t_1] -= num_c
            b_bar_tem[t_2_k][t_2] += num_c
        else:
            b_bar_tem[t_1_k][t_1] -= num
            b_bar_tem[t_2_k][t_2] += num

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b, decoding(b_bar_new, self.B_last), c_tem, min(t_1, t_2), T)
        return b_bar_tem, b_tem, c_tem

    def repair(self, b_old, b_bar_new, c, rt, T):
        b_new = np.zeros_like(b_old)
        S_tem, E_tem, A_tem = np.zeros(self.K), np.zeros(self.K), np.zeros(self.K)
        Q_tem, U_tem, R_tem, D_tem = np.zeros(self.K), np.zeros(self.K), np.zeros(self.K), np.zeros(self.K)
        S_tem[:], E_tem[:], A_tem[:] = self.S_init[:], self.E_init[:], self.A_init[:]
        Q_tem[:], U_tem[:], R_tem[:] = self.Q_init[:], self.U_init[:], self.R_init[:]
        D_tem[:] = self.D_init[:]

        for t in range(T):
            if t < rt:
                b_new[:, t] = b_old[:, t]  # 不影响前面的使用量
            else:
                for k in range(self.K):  # 当期，根据新的分配量/约束，进行使用量调整
                    cons = math.floor((U_tem[k] + A_tem[k] / self.N[k] * c[k][t]) / self.eta)
                    b_new[k][t] = min(cons, b_bar_new[k][t])  # 两者取小的（且整数）
                    b_new[k][t] = max(0, b_new[k][t])  # 并且保证非负

            S_nxt, E_nxt, A_nxt, Q_nxt, U_nxt, R_nxt, D_nxt = \
                allocation_epidemic_function(self.K, 1, S_tem, E_tem, A_tem, Q_tem, U_tem
                                             , R_tem, D_tem, self.N, self.sigma_hat, self.beta_e, self.beta_a
                                             , self.beta_u, self.alpha, self.delta_a, self.delta_q, self.delta_u
                                             , self.gamma_a, self.gamma_q, self.gamma_u, self.p, self.q
                                             , b_old[:, t:t + 1], c[:, t:t + 1], self.eta)
            S_tem[:] = S_nxt[:, 1]
            E_tem[:] = E_nxt[:, 1]
            A_tem[:] = A_nxt[:, 1]
            Q_tem[:] = Q_nxt[:, 1]
            U_tem[:] = U_nxt[:, 1]
            R_tem[:] = R_nxt[:, 1]
            D_tem[:] = D_nxt[:, 1]

        return b_new

    def difference_cal(self, b_bar_1, b_bar_2, b_1, b_2, c_1, c_2):
        b_bar_tem = np.zeros_like(self.b_bar_init)
        b_tem, c_tem = np.zeros_like(self.b_init), np.zeros_like(self.c_init)
        T = len(b_bar_1[0, :])

        up, down = 0, 0  # 向上取整、向下取整次数
        for t in range(T):
            for k in range(self.K):
                if up == down:  # 若=0，则向上取证，若=1，则向下取整
                    r = random.choice([0, 1])  # 数量相等，则随机
                elif up > down:
                    r = 1  # 哪个少，选择哪个
                else:
                    r = 0  # 哪个少，选择哪个

                b_bar_sum = b_bar_1[k][t] + b_bar_2[k][t]
                if (r == 0) and (b_bar_sum % 2 != 0):
                    up += 1
                    b_bar_tem[k][t] = math.ceil(b_bar_sum / 2)
                elif(r == 1) and (b_bar_sum % 2 != 0):
                    down += 1
                    b_bar_tem[k][t] = math.floor(b_bar_sum / 2)
                else:
                    b_bar_tem[k][t] = math.floor(b_bar_sum / 2)
            up, down = 0, 0

        up, down = 0, 0  # 向上取整、向下取整次数
        for t in range(T):
            for k in range(self.K):
                if up == down:  # 若=0，则向上取证，若=1，则向下取整
                    r = random.choice([0, 1])  # 数量相等，则随机
                elif up > down:
                    r = 1  # 哪个少，选择哪个
                else:
                    r = 0  # 哪个少，选择哪个

                c_sum = c_1[k][t] + c_2[k][t]
                if (r == 0) and (c_sum % 2 != 0):
                    up += 1
                    c_tem[k][t] = math.ceil(c_sum / 2)
                elif (r == 1) and (c_sum % 2 != 0):
                    down += 1
                    c_tem[k][t] = math.floor(c_sum / 2)
                else:
                    c_tem[k][t] = math.floor(c_sum / 2)
            up, down = 0, 0

        # 目前为止，得到了新的分配量解，以下进行修复操作，让使用量配合新的分配量，且满足约束
        b_bar_new = np.zeros_like(b_bar_tem)
        b_bar_new[:, :] = b_bar_tem[:, :]
        b_tem[:, :] = self.repair(b_1, decoding(b_bar_new, self.B_last), c_tem, 0, T)

        return b_bar_tem, b_tem, c_tem

    def picture_iter(self):
        print(f'迭代过程花费时间如下：')
        print(self.time_list)
        print(f'迭代过程结果如下：')
        print(self.fitness_list)

        y = self.fitness_list
        x = list(range(len(y)))
        plt.plot(x, y)
        plt.show()

    def roulette(self, sol_list_out, base_f):
        b_bar_tem = np.zeros_like(sol_list_out[0]['b_bar_out'])
        b_tem = np.zeros_like(sol_list_out[0]['b_out'])
        c_tem = np.zeros_like(sol_list_out[0]['c_out'])
        f_tem = 0
        tag = 0  # 是否存在提升的解

        roulette_list = [0 for s in range(len(sol_list_out))]  # 参与轮盘赌计算的是，与上一轮结果的提升差值
        for s in range(len(sol_list_out)):
            sub = sol_list_out[s]['f_out'] - base_f
            if sub > 0:
                tag += 1
                roulette_list[s] = sub
            else:
                roulette_list[s] = 0

        if tag == 0:  # 如果没有任何提升解，则直接返回，不用接下来的计算了
            return b_bar_tem, b_tem, c_tem, f_tem, tag

        p_list = [0 for s in range(len(sol_list_out))]  # 轮盘赌的累计概率表
        sum_value = sum(roulette_list)
        for s in range(len(sol_list_out)):
            if s == 0:
                p_list[s] = roulette_list[s] / sum_value
            else:
                p_list[s] = p_list[s - 1] + roulette_list[s] / sum_value

        r = random.random()
        for s in range(len(sol_list_out)):
            if r <= p_list[s] and s == 0:
                b_bar_tem[:, :] = sol_list_out[s]['b_bar_out'][:, :]
                b_tem[:, :] = sol_list_out[s]['b_out'][:, :]
                c_tem[:, :] = sol_list_out[s]['c_out'][:, :]
                f_tem = sol_list_out[s]['f_out']
            elif p_list[s - 1] < r <= p_list[s]:
                b_bar_tem[:, :] = sol_list_out[s]['b_bar_out'][:, :]
                b_tem[:, :] = sol_list_out[s]['b_out'][:, :]
                c_tem[:, :] = sol_list_out[s]['c_out'][:, :]
                f_tem = sol_list_out[s]['f_out']

        return b_bar_tem, b_tem, c_tem, f_tem, tag
