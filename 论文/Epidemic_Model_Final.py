"""
完整的传染病模型入口，包括：模型，参数拟合，多峰判断
"""
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from Metropolis_Hastings_Final import Metropolis_Hastings
from SEAQURD import SEAQURD
from Multipeak_judge import Multipeak_judge


class Epidemic_Model:  # 完整传染病模型
    def __init__(self,file_name):
        self.file_name = file_name
        self.book = load_workbook(file_name)
        self.region_num = 2

        self.sheet = []
        self.region_num = self.region_num
        for i in range(self.region_num):
            self.sheet.append(self.book[f'{i}'])  # 将每一个sheet存入列表

        # 记录完整的拟合区间（如：前60天）
        self.start = 0  # 开始时间点
        self.end = 209  # 结束时间点
        self.t_num = self.end - self.start + 1  # 时间长度

        self.actual = [[] for i in range(self.region_num)]  # 真实感染人数（从4月13号开始）
        for i in range(self.region_num):
            for j in range(self.t_num):
                self.actual[i].append(self.sheet[i].cell(1, self.start + j + 89).value)
        # print(self.actual)

        # 初始化群体，用于记录“最终”的拟合结果（即：整合了多峰后的最终结果）
        self.S = np.zeros((self.region_num, self.t_num))
        self.E = np.zeros((self.region_num, self.t_num))
        self.A = np.zeros((self.region_num, self.t_num))
        self.Q = np.zeros((self.region_num, self.t_num))
        self.U = np.zeros((self.region_num, self.t_num))
        self.R = np.zeros((self.region_num, self.t_num))
        self.D = np.zeros((self.region_num, self.t_num))

        self.I = np.zeros((self.region_num, self.t_num))

        # 总人数（不变）
        self.total_population = []
        for i in range(self.region_num):
            self.total_population.append(self.sheet[i].cell(1,2).value)
        # print(self.total_population)

        self.new_flow_node = []  # 存放新浪潮开始节点（方便画图）

        print('读取数据完毕')

        self.algorithm()  # 拟合总算法
        self.picture()

    def algorithm(self):  # 拟合总算法
        # 初始化各群体数量
        E_0 = [0 for i in range(self.region_num)]
        A_0 = [0 for i in range(self.region_num)]
        Q_0 = [0 for i in range(self.region_num)]
        U_0 = [self.actual[i][0] for i in range(self.region_num)]
        R_0 = [0 for i in range(self.region_num)]
        D_0 = [0 for i in range(self.region_num)]

        S_0 = []
        for i in range(self.region_num):
            S_0.append(self.total_population[i] - U_0[i])

        start = self.start  # 初始化拟合起点（随迭代向后推动变化）
        end = self.end  # 初始化拟合终点（始终不变）

        while True:  # 循环：不断迭代，对新浪潮进行拟合
            # （1）MCMC算法，参数拟合
            # 输入：地区数量，文件名，起点，终点，总人数，各群体初值
            # 拟合后得到：para，为参数的集合列表
            sample = Metropolis_Hastings(self.region_num,self.file_name,start,end,self.total_population
                                         ,S_0,E_0,A_0,Q_0,U_0,R_0,D_0)
            # print(sample.para)

            # （2）将拟合得到的参数，输入传染病模型，从而得到各群体的拟合数量
            # 输入：地区数量，文件名，起点，终点，总人数，各群体初值，拟合好的参数
            # 模型运算后得到：各群体拟合数量
            # 注：拟合数量的list长度，随着起点的向后推移而逐渐变短
            fitting = SEAQURD(self.region_num,self.file_name,start,end,self.total_population
                              ,S_0,E_0,A_0,Q_0,U_0,R_0,D_0,sample.para)

            # 将真实感染人数list进行切片，目的是与拟合list长度和对应区间保持一致
            # 同样也是随着起点向后变短
            act = []
            for i in range(len(self.actual)):
                act.append(self.actual[i][start:end+1])
            total_I = []
            total_S = []
            total_actual = []
            for i in range(end-start+1):
                ii = 0
                ss = 0
                aa = 0
                for j in range(self.region_num):
                    ii += fitting.I[j][i]
                    ss += fitting.S[j][i]
                    aa += act[j][i]
                total_I.append(ii)
                total_S.append(ss)
                total_actual.append(aa)
            # print(len(total_I),len(total_actual))

            # （3）多峰判断，确定变化点
            # 输入：真实感染人数list（切片后），该区间的拟合好的S数量，该区间的拟合好的I数量
            # 得到：新浪潮开始节点peak_node
            peak = Multipeak_judge(total_actual,total_S,total_I)

            for i in range(start,start+peak.peak_node+1):  # 至此，将该区间的结果存入“最终”结果中
                # 注意：“最终”list是从“起点”开始的，拟合结果直接从头开始的
                # 如：本次循环对应区间为[10,25]，则对应最终结果的[10,25],对应拟合结果的[0,15]
                for j in range(self.region_num):
                    self.S[j][i] = fitting.S[j][i-start]
                    self.E[j][i] = fitting.E[j][i-start]
                    self.A[j][i] = fitting.A[j][i-start]
                    self.Q[j][i] = fitting.Q[j][i-start]
                    self.U[j][i] = fitting.U[j][i-start]
                    self.R[j][i] = fitting.R[j][i-start]
                    self.D[j][i] = fitting.D[j][i-start]

            if peak.exist_multipeak == 'no':  # 若不存在浪潮，退出循环
                break
            else:
                # 否则记录新浪潮节点
                print(f'新浪潮节点：{start+peak.peak_node}')
                self.new_flow_node.append(start+peak.peak_node)

                # 更新 新浪潮的起点状态，就是上次浪潮的结束状态
                for i in range(self.region_num):
                    S_0[i] = fitting.S[i][peak.peak_node]
                    E_0[i] = fitting.E[i][peak.peak_node]
                    A_0[i] = fitting.A[i][peak.peak_node]
                    Q_0[i] = fitting.Q[i][peak.peak_node]
                    U_0[i] = fitting.U[i][peak.peak_node]
                    R_0[i] = fitting.R[i][peak.peak_node]
                    D_0[i] = fitting.D[i][peak.peak_node]

                start = peak.peak_node + start  # 更新起点
                continue
        self.I = self.A + self.Q + self.U

    def picture(self):
        t_range = np.arange(0, self.t_num)  # 时间跨度，分成一天份

        for t in self.new_flow_node:  # 画出浪潮分割节点
            plt.plot([t, t], [0, 50000])

        total_S, total_E, total_A, total_Q, total_U, total_R, total_D = [],[],[],[],[],[],[]
        total_I = []
        total_actual = []
        for i in range(self.t_num):
            s, e, a, q, u, r, d = 0, 0, 0, 0, 0, 0, 0
            ii = 0
            aa = 0
            for j in range(self.region_num):
                s += self.S[j][i]
                e += self.E[j][i]
                a += self.A[j][i]
                q += self.Q[j][i]
                u += self.U[j][i]
                r += self.R[j][i]
                d += self.D[j][i]
                ii += self.I[j][i]
                aa += self.actual[j][i]
            total_S.append(s), total_E.append(e), total_A.append(a), total_Q.append(q)
            total_U.append(u), total_R.append(r), total_D.append(d)
            total_I.append(ii)
            total_actual.append(aa)
        # plt.plot(t_range, total_S, label='total_S', marker='.', color='orange')
        plt.plot(t_range, total_E, label='total_E', color='red')
        plt.plot(t_range, total_A, label='total_A', color='purple')
        plt.plot(t_range, total_Q, label='total_Q', color='olivedrab')
        plt.plot(t_range, total_U, label='total_U', color='green')
        plt.plot(t_range, total_R, label='total_R', color='darkblue')
        plt.plot(t_range, total_D, label='total_D', color='black')
        plt.plot(t_range, total_I, label='total_I', color='darkred')
        plt.plot(t_range, total_actual, label='total_actual', marker='.')
        plt.legend(fontsize=10, facecolor='lightyellow')

        plt.title('SEIR Model')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('Number')
        plt.show()


if __name__ == '__main__':
    Epidemic_Model('American_data.xlsx')
