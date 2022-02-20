import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import random
import math
'''S：易感者、E：潜伏者、I：感染者、R：治愈者'''


class Metropolis_Hastings:  # Metropolis_Hastings采样算法
    def __init__(self,file_name,start,end,total_population,S_0,E_0,I_0,R_0):
        self.book = load_workbook(file_name)  # 读取数据
        self.sheet = self.book['湖北']

        self.start = start  # 开始时间点
        self.end = end  # 结束时间点
        self.t_num = self.end - self.start + 1  # 时间长度

        self.actual = []  # 真实的感染人数，读取数据即可
        for i in range(self.start,self.end+1):
            self.actual.append(self.sheet.cell(4+i,4).value)

        # 人群总数（武汉有5800w人口，全算上根本无法拟合，取10w看起来好很多）
        self.total_population = total_population
        self.S_0 = S_0
        self.E_0 = E_0
        self.I_0 = I_0
        self.R_0 = R_0

        self.iter = 100000  # 采样算法迭代次数

        self.para = []

        self.sampling()

    def sampling(self):
        beta_i = 0.35
        beta_e = 0.35
        alpha = 0.1
        gamma = 0.05

        S = [0 for i in range(self.t_num)]
        E = [0 for i in range(self.t_num)]
        I = [0 for i in range(self.t_num)]
        R = [0 for i in range(self.t_num)]
        S[0] = self.S_0
        E[0] = self.E_0
        I[0] = self.I_0
        R[0] = self.R_0

        SSE = self.algorithm(S,E,I,R,beta_i,beta_e,alpha,gamma)

        for i in range(self.iter):
            if i % 20000 == 0:
                print(f'完成迭代{i}次')
            SSE_bef = SSE

            beta_i_bef = beta_i
            beta_e_bef = beta_e
            alpha_bef = alpha
            gamma_bef = gamma

            beta_i = random.uniform(0,1)
            beta_e = random.uniform(0,1)
            alpha = random.uniform(0,0.3)
            gamma = random.uniform(0,0.2)

            SSE = self.algorithm(S,E,I,R,beta_i,beta_e,alpha,gamma)

            LR = math.exp(min(700,(-SSE + SSE_bef)))
            r = random.uniform(0,1)
            if LR >= r:
                continue
            else:
                beta_i = beta_i_bef
                beta_e = beta_e_bef
                alpha = alpha_bef
                gamma = gamma_bef
                SSE = SSE_bef
        self.para = [beta_i, beta_e, alpha, gamma]
        print(f'beta_i:{beta_i} ; beta_e:{beta_e} ; alpha:{alpha} ; gamma:{gamma}')
        print(SSE)

    def algorithm(self,S,E,I,R,beta_i,beta_e,alpha,gamma):
        for i in range(1,self.t_num):
            S[i] = S[i-1] - beta_i * I[i-1] * S[i-1] / self.total_population \
                        - beta_e * E[i-1] * S[i-1] / self.total_population

            E[i] = E[i-1] + beta_i * I[i-1] * S[i-1] / self.total_population \
                        + beta_e * E[i-1] * S[i-1] / self.total_population - alpha * E[i-1]

            I[i] = I[i-1] + alpha * E[i-1] - gamma * I[i-1]

            R[i] = R[i-1] + gamma * I[i-1]

        SSE = 0
        for i in range(self.t_num):
            SSE += (I[i] - self.actual[i]) ** 2 / 1000

        return SSE


if __name__ == '__main__':
    Metropolis_Hastings('疫情人数各省市数据统计列表.xlsx',0,59,150000,150000-41,0,41,0)
