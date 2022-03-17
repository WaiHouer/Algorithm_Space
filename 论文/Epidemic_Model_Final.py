"""
完整的传染病模型入口，包括：模型，参数拟合，多峰判断
"""
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import time
from Metropolis_Hastings_Final import Metropolis_Hastings
from SEAQURD import SEAQURD
from Multipeak_judge import Multipeak_judge


class Epidemic_Model:  # 完整传染病模型
    def __init__(self,file_name):
        self.start_time = time.time()

        self.file_name = file_name
        self.book = load_workbook(file_name)
        self.region_num = 2

        self.sheet = []
        self.region_num = self.region_num
        for i in range(self.region_num):
            self.sheet.append(self.book[f'{i}'])  # 将每一个sheet存入列表

        self.region_name = ['' for i in range(self.region_num)]  # 记录每个州的名字，便于画图
        for i in range(self.region_num):
            self.region_name[i] = self.sheet[i].cell(1,1).value

        # 记录完整的拟合区间（如：4月13日起，前self.end - self.start + 1天）
        self.start = 0  # 开始时间点
        self.end = 275  # 结束时间点（20.4.13-21.1.13，此处为275）
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

        self.observe = [0,'no']  # 设置一个观察收敛性的开关，看一次就行了，省的多峰判断，每判断一次看一次

        self.predict_num = 30  # 预测未来天数
        self.final_para = []  # 用于存放终末状态的拟合参数
        # 初始化预测阶段群体，用于记录“最终”的预测结果
        self.S_pre = np.zeros((self.region_num, self.predict_num))
        self.E_pre = np.zeros((self.region_num, self.predict_num))
        self.A_pre = np.zeros((self.region_num, self.predict_num))
        self.Q_pre = np.zeros((self.region_num, self.predict_num))
        self.U_pre = np.zeros((self.region_num, self.predict_num))
        self.R_pre = np.zeros((self.region_num, self.predict_num))
        self.D_pre = np.zeros((self.region_num, self.predict_num))
        self.I_pre = np.zeros((self.region_num, self.predict_num))
        self.actual_pre = [[] for i in range(self.region_num)]  # 真实感染人数（从终末开始）
        for i in range(self.region_num):
            for j in range(self.predict_num):
                self.actual_pre[i].append(self.sheet[i].cell(1, self.end + 1 + j + 89).value)

        self.algorithm()  # 拟合总算法
        # self.predict()  # 预测算法
        self.picture()

        self.end_time = time.time()
        print(f'SEYIAQURD-传染病模型总体运行时间为：{self.end_time - self.start_time}')

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

        s = self.start  # 初始化拟合起点（随迭代向后推动变化）
        e = self.end  # 初始化拟合终点（始终不变）
        start_list = []
        end_list = []
        while e - s + 1 >= 30:  # 每30天为一个拟合周期，这样拟合更精准且迭代次数更少（不足2倍周期的看做一个周期）
            start_list.append(s)
            end_list.append(s + 14)
            s += 14
        if s != e:  # 若有剩余，代表最后不足2倍周期，则看做最后一个周期
            start_list.append(s)
            end_list.append(e)

        for st in range(len(start_list)):  # 对每个周期进行拟合
            start = start_list[st]
            end = end_list[st]

            self.observe[0] += 1  # 观察拟合参数的周期计数+1

            while True:  # 循环：不断迭代，对新浪潮进行拟合
                # （1）MCMC算法，参数拟合
                # 输入：地区数量，文件名，起点，终点，总人数，各群体初值
                # 拟合后得到：para，为参数的集合列表
                sample = Metropolis_Hastings(self.region_num,self.file_name,start,end,self.total_population
                                             ,S_0,E_0,A_0,Q_0,U_0,R_0,D_0)
                # print(sample.para)

                if self.observe[0] == 8 and self.observe[1] == 'no':  # 观察一次（只观察第n段的拟合情况）
                    # print(sample.observe_para)
                    self.observe_para(sample.observe_para)
                    self.observe[1] = 'yes'

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

                for i in range(start,start+peak.peak_node+1):  # 至此，将该区间的结果存入“最终”结果中（修改：+1）
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
                    self.final_para = sample.para  # 记录一下最后阶段的拟合参数，用于预测未来
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

            for i in range(self.region_num):  # 更新下一个周期的起始状态，即本周期的结束状态
                S_0[i] = self.S[i][end]
                E_0[i] = self.E[i][end]
                A_0[i] = self.A[i][end]
                Q_0[i] = self.Q[i][end]
                U_0[i] = self.U[i][end]
                R_0[i] = self.R[i][end]
                D_0[i] = self.D[i][end]

        self.I = self.A + self.Q + self.U

    def picture(self):
        t_range = np.arange(0, self.t_num)  # 时间跨度，分成一天份
        t_pre_range = np.arange(self.end + 1, self.end + 1 + self.predict_num)

        for t in self.new_flow_node:  # 画出浪潮分割节点
            plt.plot([t, t], [0, 100000])

        # 计算出各区域加总在一起的总S、E、A、Q、U、R、D、感染人数、真实人数，并画出图像
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
        # plt.plot(t_range, total_S, label='Total_S', marker='.', color='orange')  # 群体过大
        # plt.plot(t_range, total_E, label='Total_E', color='red')  # 群体过大
        # plt.plot(t_range, total_A, label='Total_A', color='purple')
        # plt.plot(t_range, total_Q, label='Total_Q', color='olivedrab')
        # plt.plot(t_range, total_U, label='Total_U', color='green')
        # plt.plot(t_range, total_R, label='Total_R', color='darkblue')
        # plt.plot(t_range, total_D, label='Total_D', color='black')
        plt.plot(t_range, total_I, label='Total_I', color='darkred')
        if self.region_num > 1:
            for i in range(self.region_num):
                plt.plot(t_range, self.I[i], label=f'{self.region_name[i]}_I')
                plt.plot(t_range, self.actual[i], label=f'{self.region_name[i]}_Actual', marker='.')
        plt.plot(t_range, total_actual, label='Total_Actual', color='yellowgreen', marker='.')
        plt.legend(fontsize=10, facecolor='lightyellow')

        total_S_pre, total_E_pre, total_A_pre, total_Q_pre, total_U_pre, total_R_pre, total_D_pre = [], [], [], [], [], [], []
        total_I_pre = []
        total_actual_pre = []
        for i in range(self.predict_num):
            s, e, a, q, u, r, d = 0, 0, 0, 0, 0, 0, 0
            ii = 0
            aa = 0
            for j in range(self.region_num):
                s += self.S_pre[j][i]
                e += self.E_pre[j][i]
                a += self.A_pre[j][i]
                q += self.Q_pre[j][i]
                u += self.U_pre[j][i]
                r += self.R_pre[j][i]
                d += self.D_pre[j][i]
                ii += self.I_pre[j][i]
                aa += self.actual_pre[j][i]
            total_S_pre.append(s), total_E_pre.append(e), total_A_pre.append(a), total_Q_pre.append(q)
            total_U_pre.append(u), total_R_pre.append(r), total_D_pre.append(d)
            total_I_pre.append(ii)
            total_actual_pre.append(aa)
        # 预测只简单画了总人数延长，没有画各区域，没有滚动
        # plt.plot(t_pre_range, total_actual_pre, label='Total_Actual_pre', color='yellowgreen', marker='.')
        # plt.plot(t_pre_range, total_I_pre, label='Total_I_pre', color='salmon', marker='*')

        month_num = [0,30,61,91,122,153,183,214,244,275]
        month = ['4/13/20','5/13/20','6/13/20','7/13/20','8/13/20','9/13/20','10/13/20','11/13/20','12/13/20','1/13/21']
        plt.xticks(month_num, month)

        plt.title('SEYIAQURD Model')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('Number')
        plt.show()

    def observe_para(self,observe_para):
        x = np.arange(0, len(observe_para[0][0]))
        for i in range(self.region_num):
            plt.plot(x, observe_para[i][0], label='beta_e')
            plt.plot(x, observe_para[i][1], label='beta_a')
            plt.plot(x, observe_para[i][2], label='beta_u')
            plt.plot(x, observe_para[i][3], label='alpha')
            plt.plot(x, observe_para[i][4], label='delta_a')
            plt.plot(x, observe_para[i][5], label='delta_q')
            plt.plot(x, observe_para[i][6], label='delta_u')
            plt.plot(x, observe_para[i][7], label='gamma_a')
            plt.plot(x, observe_para[i][8], label='gamma_q')
            plt.plot(x, observe_para[i][9], label='gamma_u')
            plt.title(f'Observe Parameters - {self.region_name[i]}')
            plt.legend(fontsize=10, facecolor='lightyellow')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Parameters')
            plt.show()

    def predict(self):
        S_0 = [0 for i in range(self.region_num)]
        E_0 = [0 for i in range(self.region_num)]
        A_0 = [0 for i in range(self.region_num)]
        Q_0 = [0 for i in range(self.region_num)]
        U_0 = [0 for i in range(self.region_num)]
        R_0 = [0 for i in range(self.region_num)]
        D_0 = [0 for i in range(self.region_num)]

        for i in range(self.region_num):  # 取出最后的终末状态最为预测的初始状态
            S_0[i] = self.S[i][-1]
            E_0[i] = self.E[i][-1]
            A_0[i] = self.A[i][-1]
            Q_0[i] = self.Q[i][-1]
            U_0[i] = self.U[i][-1]
            R_0[i] = self.R[i][-1]
            D_0[i] = self.D[i][-1]

        fitting = SEAQURD(self.region_num, self.file_name, self.end + 1, self.end + 1 + self.predict_num - 1,
                          self.total_population, S_0, E_0, A_0, Q_0, U_0, R_0, D_0, self.final_para)

        self.S_pre, self.E_pre, self.A_pre = fitting.S, fitting.E, fitting.A
        self.Q_pre, self.U_pre, self.R_pre, self.D_pre = fitting.Q, fitting.U, fitting.R, fitting.D

        self.I_pre = self.A_pre + self.Q_pre + self.U_pre


if __name__ == '__main__':
    Epidemic_Model('American_data.xlsx')
