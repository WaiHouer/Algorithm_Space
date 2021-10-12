'''基本模拟退火算法'''
import random
import math
import numpy as np
'''例子：
优化问题 —— max f(x1,x2)=21.5+x1*sin(4πx1)+x2*sin(20πx2)
                   s.t.  -3.0 <= x1 <= 12.1
                          4.1 <= x2 <=  5.8
'''


class Simulated_Annealing:  # 定义“模拟退火算法”类
    def __init__(self,x1_range,x2_range,tag='min'):  # 方法：初始化算法
        # 传入：x1取值范围，x2取值范围，问题类型
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.tag = tag
        self.T_max = 10000  # 初始退火温度
        self.T_min = 1     # 退火截止温度
        self.T_iteration = 1000  # 同温迭代次数
        self.rate = 0.95    # 退火速率

        self.final_sol = []   # 初始化最终解
        self.final_f = None   # 初始化最终目标函数值

        self.algorithm()  # 自动运行算法
        self.display()    # 自动显示结果

    def initial_sol(self):  # 方法：随机生成初始解
        x1 = random.uniform(self.x1_range[0],self.x1_range[1])
        x2 = random.uniform(self.x2_range[0],self.x2_range[1])
        return x1,x2

    def function(self,x1,x2):  # 方法：计算目标函数值（能量值）
        # 若为最小化问题：正常求
        # 若为最大化问题：取负，转变为最小化问题
        if self.tag == 'min':
            f = 21.5 + x1 * math.sin(4*math.pi*x1) + x2 * math.sin(20*math.pi*x2)
        else:
            f = -(21.5 + x1 * math.sin(4*math.pi*x1) + x2 * math.sin(20*math.pi*x2))

        return f

    def new_sol(self,x1,x2):  # 方法：在邻域生成新解
        delta_x1 = random.random() * 2 - 1
        delta_x2 = random.random() * 2 - 1

        if self.x1_range[0] <= x1 + delta_x1 <= self.x1_range[1]:
            x1_new = x1 + delta_x1
        else:
            x1_new = x1 - delta_x1

        if self.x2_range[0] <= x2 + delta_x2 <= self.x2_range[1]:
            x2_new = x2 + delta_x2
        else:
            x2_new = x2 - delta_x2

        return x1_new,x2_new

    def algorithm(self):  # 方法：模拟退火算法主程序
        x1,x2 = self.initial_sol()  # （1）生成初始解

        T = self.T_max
        while T > self.T_min:  # 从初始温度开始，进行循环退火
            for i in range(self.T_iteration):  # 同温迭代
                # 计算旧解的能量值
                f_old = self.function(x1,x2)

                # 在邻域生成新解
                x1_new,x2_new = self.new_sol(x1,x2)

                # 计算新解的能量值，并计算差值
                f_new = self.function(x1_new,x2_new)
                delta_f = f_new - f_old

                if delta_f < 0:  # 若新解 更小 ，则接受新解
                    x1 = x1_new
                    x2 = x2_new
                else:
                    # 否则，以 p概率 接受这个 更差的新解
                    p = np.exp(-delta_f / T)
                    if random.random() < p:
                        x1 = x1_new
                        x2 = x2_new

            T = T * self.rate  # 退火

        # 记录最终解以及能量值
        self.final_sol.append(x1)
        self.final_sol.append(x2)
        self.final_f = self.function(x1,x2)

    def display(self):
        print(f'初始退火温度：{self.T_max}')
        print(f'退火截止温度：{self.T_min}')
        print(f'同温度迭代次数：{self.T_iteration}')
        print(f'降温速率：{self.rate}')
        print(f'求得最终解为：{self.final_sol}')
        if self.tag == 'min':
            print(f'最小化，目标函数值：{self.final_f}')
        else:
            print(f'最大化，目标函数值：{-self.final_f}')


if __name__ == '__main__':
    Simulated_Annealing([-3.0,12.1],[4.1,5.8],'max')
