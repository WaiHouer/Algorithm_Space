"""
解决超大规模整数规划问题——列生成算法（Column Generation）
"""
'''
以经典的 Cutting Stock Problem 为例：
工厂有标准化生产的长木卷，每个长度为 W ，一共有 m 个顾客，每位顾客需要 ni 个长度为 wi 的小卷（i = 1,2,...,m），怎么切浪费最少

若建模成经典模型，求解不了，所以采取以下建模方式：
Xj = 第j种“切割方式”的所用次数，（假设我们所有方式都是已知的，只不过没有全部列出来）
aij = 在第j种中，长度为wi的小卷出现的次数（例：第一种方式中，w1有1卷，w2有0卷，w3有2卷，则有 a11=1;a21=0;a31=2）
模型：
min ∑j Xj
s.t. ∑j aij >= ni , i=1,2,...,m  （满足客户需求）
     Xj ∈ Z+ , j=1,2,...,n       （切割方式出现次数为正整数）
     
     其中，∑j aij的矩阵中，每一列代表一种切割方式，拥有隐含的条件：∑i aij * wi <= W （每种方式中小卷不超过大卷长度上限）

求这个问题，就要先求其线性规划松弛问题，即  Xj ∈ R+
'''
import math
from scipy.optimize import linprog
from Branch_and_Bound import Branch_and_Bound


def zero_list(m,n):  # 外部方法：用于建立m*n的 全0 list矩阵
    ll = []
    for i in range(m):
        ll.append([])
        for j in range(n):
            ll[i].append(0)
    return ll


class Column_Generation:  # 定义“列生成算法”类
    def __init__(self,max_width,demand,w_width):  # 方法：初始化
        self.max_width = max_width  # 大木卷的标准长度
        self.demand = demand        # 小木卷需求矩阵
        self.w_width = w_width      # 小木卷长度矩阵

        self.m = len(demand)        # LPM问题系数矩阵的行数

        self.cut_method = zero_list(self.m,self.m)  # 初始化切割方案矩阵（即A系数矩阵）
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    # 采取笔记中的方式，初始化切割方案（只切一种）
                    self.cut_method[i][j] = math.floor(self.max_width / self.w_width[i])

        self.lpm_x = None    # 记录LPM问题解
        self.lpm_obj = None  # 记录LPM问题目标函数值

        self.algorithm()  # 算法主体

    def lpm_solver(self):  # 方法：LPM及其对偶问题求解（返回对偶解）
        # 第一部分：计算LPM问题的解和目标函数值（目的：方便最后输出结果，实际上在循环中没有其他用处）
        n = len(self.cut_method[0])  # 系数矩阵的列数
        c = [1] * n
        lpm_ub = zero_list(len(self.cut_method),len(self.cut_method[0]))
        for i in range(len(self.cut_method)):
            for j in range(len(self.cut_method[0])):
                lpm_ub[i][j] = self.cut_method[i][j] * (-1)
        lpm_b_ub = [x * (-1) for x in self.demand]
        x_bounds = [[0,None]] * n
        bounds = ()
        for i in range(n):
            bounds = bounds + (x_bounds[i],)
        lpm_sol = linprog(c,lpm_ub,lpm_b_ub,None,None,bounds)
        self.lpm_x = lpm_sol.x      # 更新解
        self.lpm_obj = lpm_sol.fun  # 更新目标函数值
        # print(c,lpm_ub,lpm_b_ub,bounds)
        # print(lpm_sol.success,lpm_sol.x,lpm_sol.fun)

        # 第二部分：转换成对偶问题，并进行求解
        dual_n = len(self.demand)
        dual_c = [x * (-1) for x in self.demand]
        dual_ub = list(map(list, zip(*self.cut_method)))
        dual_b_ub = c
        dual_x_bounds = [[0,None]] * dual_n
        dual_bounds = ()
        for i in range(dual_n):
            dual_bounds = dual_bounds + (dual_x_bounds[i],)
        dual_sol = linprog(dual_c,dual_ub,dual_b_ub,None,None,dual_bounds)
        # print(dual_sol.success,dual_sol.x,-dual_sol.fun)
        # print('-----')

        return dual_sol.x  # 返回对偶问题的解

    def sub_problem(self,dual_x):  # 辅助问题求解（得到新切割方案）
        # 传入对偶解
        n = len(dual_x)  # 系数矩阵的列数
        c = [x * (-1) for x in dual_x]
        a_ub = [self.w_width]
        b_ub = [self.max_width]
        x_bounds = [[0,None]] * n
        bounds = ()
        for i in range(n):  # 注意记一下这里元组的加法，需要 逗号
            bounds = bounds + (x_bounds[i],)

        # 背包问题，所以用分支定界（自己写的，超级垃圾，所以千万别尝试较大的算例。。。）
        return Branch_and_Bound(c,a_ub,b_ub,None,None,bounds).final_set

    def algorithm(self):  # 方法：算法主体
        # 循环，直到找不到新的、可以改进的切割方式
        while True:
            # （1）求解LPM以及对偶，记录LPM解，并得到对偶解
            dual_x = self.lpm_solver()

            # 由对偶解形成辅助问题，求解，得到新的切割方式，得到被减去的部分的值
            new_method,alpha = self.sub_problem(dual_x)
            alpha = -round(alpha,3)
            # 注意：最大化问题，取个负号
            # 并且由于求解器的原因，可能得到1.00000000000231这种数字，但实际上应该是1，取个三位小数就行了，不然计算不准

            # 计算判断数（reduce cost），判断是否可加，循环
            if 1 - alpha >= 0:  # 不可改进，退出循环
                break
            else:               # 可改进，继续循环
                for i in range(len(new_method)):
                    self.cut_method[i].append(new_method[i])  # 加一列
                continue

        # 求得LPM解，向上取整即可得到原来LP问题整数解
        final_x = []
        for i in self.lpm_x:
            final_x.append(math.ceil(round(i,3)))
        final_obj = sum(final_x)
        print(f'方案矩阵为（每一列为一个方案）：{self.cut_method}')
        print('最终解：')
        for i in range(len(self.lpm_x)):
            print(f'方案{i}:{final_x[i]}次')
        print(f'切割木卷总数量为：{final_obj}')


if __name__ == '__main__':
    Column_Generation(218,[44,3,48],[81,70,68])
    Column_Generation(176,[33,7,40],[30,63,22])
