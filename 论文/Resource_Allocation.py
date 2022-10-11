"""
资源分配算法的入口，可以通过修改 tag 选择不同的算法
"""
import math
import time
from openpyxl import load_workbook
import numpy as np
from numba import njit
from Allocation_Epidemic_Function import allocation_epidemic_function
from Myopic_LP import myopic_lp
from Benchmark import benchmark
from ADP import adp


np.set_printoptions(suppress=True)


class Resource_Allocation:
    def __init__(self):
        self.K = 9  # 地区数量
        self.T = 28  # 资源分配时间跨度
        self.sigma_hat = np.zeros((self.K, self.K))  # 区域之间的地理系数，矩阵
        '---------------------------------'
        self.book = load_workbook('Fitting_Result.xlsx')
        self.group_initial = self.book[f'群体']
        self.para_initial = self.book[f'参数']
        self.info = self.book[f'区域信息']

        self.location = np.zeros((self.K, 2))  # 存放坐标
        self.N = np.zeros(self.K)  # 存放人口
        self.name = []  # 存放名字
        for i in range(self.K):
            self.N[i] = self.info.cell(2, i + 2).value
            self.name.append(self.info.cell(1, i + 2).value)
            for j in range(2):
                self.location[i][0] = self.info.cell(3, i + 2).value
                self.location[i][1] = self.info.cell(4, i + 2).value
        '---------------------------------'
        self.S_initial = np.zeros(self.K)  # 群体，初始条件
        self.E_initial = np.zeros(self.K)
        self.A_initial = np.zeros(self.K)
        self.Q_initial = np.zeros(self.K)
        self.U_initial = np.zeros(self.K)
        self.R_initial = np.zeros(self.K)
        self.D_initial = np.zeros(self.K)
        self.beta_e, self.beta_a, self.beta_u = np.zeros(self.K), np.zeros(self.K), np.zeros(self.K)  # 参数，初始条件
        self.alpha = np.zeros(self.K)
        self.delta_a, self.delta_q, self.delta_u = np.zeros(self.K), np.zeros(self.K), np.zeros(self.K)
        self.gamma_a, self.gamma_q, self.gamma_u = np.zeros(self.K), np.zeros(self.K), np.zeros(self.K)
        self.p, self.q = self.para_initial.cell(11, 2).value, self.para_initial.cell(12, 2).value
        self.sigma = self.para_initial.cell(13, 2).value
        for k in range(self.K):
            self.S_initial[k] = self.group_initial.cell(1, k + 2).value
            self.E_initial[k] = self.group_initial.cell(2, k + 2).value
            self.A_initial[k] = self.group_initial.cell(3, k + 2).value
            self.Q_initial[k] = self.group_initial.cell(4, k + 2).value
            self.U_initial[k] = self.group_initial.cell(5, k + 2).value
            self.R_initial[k] = self.group_initial.cell(6, k + 2).value
            self.D_initial[k] = self.group_initial.cell(7, k + 2).value
            self.beta_e[k] = self.para_initial.cell(1, k + 2).value
            self.beta_a[k] = self.para_initial.cell(2, k + 2).value
            self.beta_u[k] = self.para_initial.cell(3, k + 2).value
            self.alpha[k] = self.para_initial.cell(4, k + 2).value
            self.delta_a[k] = self.para_initial.cell(5, k + 2).value
            self.delta_q[k] = self.para_initial.cell(6, k + 2).value
            self.delta_u[k] = self.para_initial.cell(7, k + 2).value
            self.gamma_a[k] = self.para_initial.cell(8, k + 2).value
            self.gamma_q[k] = self.para_initial.cell(9, k + 2).value
            self.gamma_u[k] = self.para_initial.cell(10, k + 2).value
        self.sigma_hat = d_m_calculate(self.K, self.location, self.sigma)
        '---------------------------------'
        self.S_initial_last = np.zeros(self.K)  # 拟合的倒数第二个时期的群体，即：初始条件前一个时期
        self.E_initial_last = np.zeros(self.K)  # 用于计算U_new
        self.A_initial_last = np.zeros(self.K)
        self.Q_initial_last = np.zeros(self.K)
        self.U_initial_last = np.zeros(self.K)
        self.R_initial_last = np.zeros(self.K)
        self.D_initial_last = np.zeros(self.K)
        for k in range(self.K):
            self.S_initial_last[k] = self.group_initial.cell(11, k + 2).value
            self.E_initial_last[k] = self.group_initial.cell(12, k + 2).value
            self.A_initial_last[k] = self.group_initial.cell(13, k + 2).value
            self.Q_initial_last[k] = self.group_initial.cell(14, k + 2).value
            self.U_initial_last[k] = self.group_initial.cell(15, k + 2).value
            self.R_initial_last[k] = self.group_initial.cell(16, k + 2).value
            self.D_initial_last[k] = self.group_initial.cell(17, k + 2).value
        '---------------------------------'
        #  注：所有时间相关都用T+1，因为要把初期算上
        self.eta = 0.5  # 病床有效率
        self.b_hat = np.zeros(self.T + 1)  # 每期新增病床（注意别太大，容易问题不可行：b_last>U，累计病床过剩问题！！）
        for i in range(self.T + 1):
            self.b_hat[i] = 700
        self.lambda_b = 0.15  # 新增病床部署率（防止只发放给某一个区域）
        self.C = np.zeros(self.T + 1)  # 每期新增核酸检测（有几个区域系数为正，则不会分配给该区域，即使资源浪费）
        for i in range(self.T + 1):
            self.C[i] = 500000
        self.lambda_c = 0.15  # 新增核酸检测部署率（防止只发放给某一个区域）
        self.M = 20  # ADP算法，迭代修正次数
        self.L = 80  # ADP算法，单次修正，样本量
        self.O = 5 * self.K  # 状态变量（特征值）数量（没有参与任何计算，仅供展示）
        self.W = 30  # 神经元个数
        self.delta = 999999999999  # 足够大的正数（没有参与任何计算，仅供展示）
        '---------------------------------'

        self.algorithm()

    def algorithm(self):
        tag = 'all'

        value_nature = 0  # 自然状态下的总新增
        if tag:
            s_time = time.time()
            b, c = np.zeros((self.K, self.T + 1 + 1)), np.zeros((self.K, self.T + 1 + 1))  # 为了算出value值，额外多迭代一期
            S, E, A, Q, U, R, D = allocation_epidemic_function(self.K, self.T + 1, self.S_initial, self.E_initial
                                                               , self.A_initial, self.Q_initial, self.U_initial
                                                               , self.R_initial, self.D_initial, self.N, self.sigma_hat
                                                               , self.beta_e, self.beta_a, self.beta_u, self.alpha
                                                               , self.delta_a, self.delta_q, self.delta_u
                                                               , self.gamma_a, self.gamma_q, self.gamma_u
                                                               , self.p, self.q, b, c, self.eta)
            e_time = time.time()
            for t in range(self.T + 1):
                for k in range(self.K):
                    value_nature += E[k][t + 1] - E[k][t] + self.alpha[k] * E[k][t]
            print(value_nature)
            print(f'自然状态迭代时间：{e_time - s_time}s')

        if tag == 'Myopic_LP' or tag == 'all':
            s_time = time.time()
            b_myopic, c_myopic, value_myopic = myopic_lp(self.K, self.T, self.S_initial, self.E_initial
                                                         , self.A_initial, self.Q_initial, self.U_initial
                                                         , self.R_initial, self.D_initial, self.N, self.sigma_hat
                                                         , self.beta_e, self.beta_a, self.beta_u, self.alpha
                                                         , self.delta_a, self.delta_q, self.delta_u
                                                         , self.gamma_a, self.gamma_q, self.gamma_u
                                                         , self.p, self.q, self.eta, self.b_hat, self.lambda_b
                                                         , self.C, self.lambda_c)
            e_time = time.time()
            print(sum(value_myopic),'=>',f'减少新增：{value_nature - sum(value_myopic)}'
                                         f'，百分比：{(value_nature - sum(value_myopic)) / value_nature}')
            print(f'Myopic_LP算法运行时间：{e_time - s_time}s')

        if tag == 'Benchmark_Average' or tag == 'all':
            s_time = time.time()
            b_BH_Aver, c_BH_Aver, value_BH_Aver = benchmark(self.K, self.T, self.S_initial, self.E_initial
                                                            , self.A_initial, self.Q_initial, self.U_initial
                                                            , self.R_initial, self.D_initial, self.N, self.sigma_hat
                                                            , self.beta_e, self.beta_a, self.beta_u, self.alpha
                                                            , self.delta_a, self.delta_q, self.delta_u
                                                            , self.gamma_a, self.gamma_q, self.gamma_u
                                                            , self.p, self.q, self.eta, self.b_hat, self.C
                                                            , 'Benchmark_Average', np.zeros(self.K))
            e_time = time.time()
            print(sum(value_BH_Aver), '=>', f'减少新增：{value_nature - sum(value_BH_Aver)}'
                                            f'，百分比：{(value_nature - sum(value_BH_Aver)) / value_nature}')
            print(f'Benchmark_Average算法运行时间：{e_time - s_time}s')

        if tag == 'Benchmark_N' or tag == 'all':
            s_time = time.time()
            b_BH_N, c_BH_N, value_BH_N = benchmark(self.K, self.T, self.S_initial, self.E_initial
                                                   , self.A_initial, self.Q_initial, self.U_initial
                                                   , self.R_initial, self.D_initial, self.N, self.sigma_hat
                                                   , self.beta_e, self.beta_a, self.beta_u, self.alpha
                                                   , self.delta_a, self.delta_q, self.delta_u
                                                   , self.gamma_a, self.gamma_q, self.gamma_u
                                                   , self.p, self.q, self.eta, self.b_hat, self.C
                                                   , 'Benchmark_N', np.zeros(self.K))
            e_time = time.time()
            print(sum(value_BH_N), '=>', f'减少新增：{value_nature - sum(value_BH_N)}'
                                         f'，百分比：{(value_nature - sum(value_BH_N)) / value_nature}')
            print(f'Benchmark_N算法运行时间：{e_time - s_time}s')

        if tag == 'Benchmark_U' or tag == 'all':
            s_time = time.time()
            b_BH_U, c_BH_U, value_BH_U = benchmark(self.K, self.T, self.S_initial, self.E_initial
                                                   , self.A_initial, self.Q_initial, self.U_initial
                                                   , self.R_initial, self.D_initial, self.N, self.sigma_hat
                                                   , self.beta_e, self.beta_a, self.beta_u, self.alpha
                                                   , self.delta_a, self.delta_q, self.delta_u
                                                   , self.gamma_a, self.gamma_q, self.gamma_u
                                                   , self.p, self.q, self.eta, self.b_hat, self.C
                                                   , 'Benchmark_U', np.zeros(self.K))
            e_time = time.time()
            print(sum(value_BH_U), '=>', f'减少新增：{value_nature - sum(value_BH_U)}'
                                         f'，百分比：{(value_nature - sum(value_BH_U)) / value_nature}')
            print(f'Benchmark_U算法运行时间：{e_time - s_time}s')

        if tag == 'Benchmark_U_new' or tag == 'all':
            s_time = time.time()
            b_BH_U_n, c_BH_U_n, value_BH_U_n = benchmark(self.K, self.T, self.S_initial, self.E_initial
                                                         , self.A_initial, self.Q_initial, self.U_initial
                                                         , self.R_initial, self.D_initial, self.N, self.sigma_hat
                                                         , self.beta_e, self.beta_a, self.beta_u, self.alpha
                                                         , self.delta_a, self.delta_q, self.delta_u
                                                         , self.gamma_a, self.gamma_q, self.gamma_u
                                                         , self.p, self.q, self.eta, self.b_hat, self.C
                                                         , 'Benchmark_U_new', np.zeros(self.K), self.E_initial_last)
            e_time = time.time()
            print(sum(value_BH_U_n), '=>', f'减少新增：{value_nature - sum(value_BH_U_n)}'
                                           f'，百分比：{(value_nature - sum(value_BH_U_n)) / value_nature}')
            print(f'Benchmark_U_new算法运行时间：{e_time - s_time}s')

        if tag == 'ADP' or tag == 'all':
            s_time = time.time()
            b_ADP, c_ADP, value_ADP = adp(self.K, self.T, self.S_initial, self.E_initial
                                          , self.A_initial, self.Q_initial, self.U_initial
                                          , self.R_initial, self.D_initial, self.N, self.sigma_hat
                                          , self.beta_e, self.beta_a, self.beta_u, self.alpha
                                          , self.delta_a, self.delta_q, self.delta_u
                                          , self.gamma_a, self.gamma_q, self.gamma_u
                                          , self.p, self.q, self.eta, self.b_hat, self.C, self.E_initial_last
                                          , self.M, self.L, self.lambda_b, self.lambda_c, self.W)
            e_time = time.time()
            print(sum(value_ADP), '=>', f'减少新增：{value_nature - sum(value_ADP)}'
                                        f'，百分比：{(value_nature - sum(value_ADP)) / value_nature}')
            print(f'ADP算法运行时间：{e_time - s_time}s')


@njit()
def d_m_calculate(K, location, c_0):  # 计算m值
    m = np.zeros((K, K))
    d = np.zeros((K, K))
    for k in range(K):
        for j in range(K):  # 计算地理距离的倒数
            if k != j:
                d[k][j] = 1 / math.sqrt((location[k][0] - location[j][0]) ** 2 +
                                        (location[k][1] - location[j][1]) ** 2)
    for k in range(K):
        for j in range(K):  # 计算地理系数
            if k == j:
                m[k][j] = c_0
            else:
                m[k][j] = (1 - c_0) * d[k][j] / sum(d[:, j])
    return m


if __name__ == '__main__':
    Resource_Allocation()
