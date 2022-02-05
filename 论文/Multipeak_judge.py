'''
用于判断多峰起始点
'''
from SEIR import SEIR
import math


class Multipeak_judge:
    def __init__(self,file_name):
        self.Model = SEIR(file_name)

        self.t_num = self.Model.t_num
        self.actual = self.Model.actual
        self.I = self.Model.I
        self.S = self.Model.S

        self.start_point = 1

        self.m_s = 2.5
        self.m_0 = 500
        self.m_threshold = 505

        # 由于要计算新增，是做差，所以此两个list第一个元素为None
        self.i_hat = [None for t in range(self.t_num)]
        self.i_act = [None for t in range(self.t_num)]
        for t in range(self.start_point,self.t_num):
            self.i_hat[t] = self.I[t] - self.I[t-1]
            self.i_act[t] = self.actual[t] - self.actual[t-1]
        print(self.i_act)
        print(self.i_hat)

        self.z = [None for t in range(self.t_num)]
        self.p = [None for t in range(self.t_num)]
        self.m = [None for t in range(self.t_num)]

        self.z_calculate()
        self.p_calculate()
        self.m_calculate()
        # 目前有点问题：
        # （1）不应该一口气全算出来z，p，m，因为涉及到重置问题，不一样（改一下即可）
        # （2）没加m的下限值（加上即可）

        self.Model.picture()

    def z_calculate(self):
        for t in range(self.start_point,self.t_num):
            self.z[t] = math.fabs(self.i_hat[t] - self.i_act[t]) / math.sqrt(self.S[t] * self.I[t])
        print(self.z)

    def p_calculate(self):
        for t in range(self.start_point + 1,self.t_num):
            larger_tem = 0
            for u in range(self.start_point,t):
                if self.z[u] >= self.z[t]:
                    larger_tem += 1
            self.p[t] = larger_tem / (t - self.start_point)
        print(self.p)

    def m_calculate(self):
        for t in range(self.start_point,self.t_num):
            if t == self.start_point:
                self.m[t] = self.m_0
            else:
                self.m[t] = self.m[t-1] * (self.m_s / (1 - math.exp(- self.m_s))) * math.exp(- self.m_s * self.p[t])
                print(self.p[t],math.exp(- self.m_s * self.p[t]))

            if self.m[t] > self.m_threshold:
                print(t,self.m)
                break


if __name__ == '__main__':
    Multipeak_judge('疫情人数各省市数据统计列表.xlsx')
