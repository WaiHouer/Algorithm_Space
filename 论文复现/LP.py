"""
Myopic_LP算法中的整数规划模型部分
"""
import numpy as np
import pulp as pl


class LP:
    def __init__(self,region_num,total_population,B_before,a_before,b,r,ksi,para,dist,S,A,U):
        self.region_num = region_num
        self.total_population = total_population

        self.B_before = B_before
        self.a_before = a_before

        self.b = b
        self.r = r
        self.ksi = ksi

        self.para = para
        self.beta_e, self.beta_a, self.beta_u, self.alpha, self.delta_a, self.delta_q, self.delta_u, \
        self.gamma_a, self.gamma_q, self.gamma_u, self.p, self.q, self.c_0 = self.para

        self.dist = dist
        self.m = np.zeros((self.region_num, self.region_num))
        self.m_calculate()

        self.S, self.A, self.U = S, A, U

        self.a = [0 for i in range(self.region_num)]  # 存放本次优化的变量值
        self.B = 0  # 存放本次病床B

        self.algorithm()
        print(self.a)
        print(self.B)

    def algorithm(self):
        # 初始化问题（最小化问题）
        problem = pl.LpProblem('LP', sense=pl.LpMinimize)
        print(f'开始求解问题：{problem.name}')

        # 规定变量
        variables = [pl.LpVariable(f'a{i}', lowBound=0, cat=pl.LpInteger) for i in range(self.region_num)]
        print(variables)

        # 规定目标函数
        c = []
        for j in range(self.region_num):
            S_m_sum = 0
            for i in range(self.region_num):
                S_m_sum += self.S[i] * self.m[i][j]
            c_tem = - (self.beta_a[j] + self.beta_u[j]) / self.total_population[j] * S_m_sum
            c.append(c_tem)
        print(f'目标函数系数向量：{c}')

        objective = sum([c[i] * variables[i] for i in range(self.region_num)])
        print(f'目标函数：{objective}')

        # 规定约束
        constraints = []

        st_1 = [1 for i in range(self.region_num)]  # （1）可用床位约束 + 未使用床位顺延
        right_1 = self.B_before
        # for i in range(self.region_num):
        #     right_1 -= self.a_before[i]
        right_1 += self.b
        constraints.append(sum([st_1[i] * variables[i] for i in range(self.region_num)]) <= right_1)
        self.B = right_1  # 存放本次的病床B

        for i in range(self.region_num):  # 床位小于感染者数量
            st_2 = [0 for k in range(i)] + [1] + [0 for j in range(self.region_num - i - 1)]
            constraints.append(sum([st_2[j] * variables[j] for j in range(self.region_num)]) <= self.A[i] + self.U[i])

        for i in range(self.region_num):  # 已分配床位不变，顺延
            st_2 = [0 for k in range(i)] + [1] + [0 for j in range(self.region_num - i - 1)]
            constraints.append(sum([st_2[j] * variables[j] for j in range(self.region_num)]) >= self.a_before[i])

        for i in range(self.region_num):  #
            st_2 = [0 for k in range(i)] + [1] + [0 for j in range(self.region_num - i - 1)]
            constraints.append(sum([st_2[j] * variables[j] for j in range(self.region_num)]) <= self.a_before[i] + self.r)

        print(f'约束条件是：')
        for i in range(len(constraints)):
            print(constraints[i])

        # 整合并求解
        problem += objective
        for i in constraints:
            problem += i
        problem.solve()
        # print("Shan Status:", pl.LpStatus[problem.status])  # 输出求解状态
        t = 0
        for v in problem.variables():
            self.a[t] = v.varValue
            t += 1
            print(v.name, "=", v.varValue)  # 输出每个变量的最优值
        print("F(x) =", pl.value(problem.objective))  # 输出最优解的目标函数值


    def m_calculate(self):
        for i in range(self.region_num):
            for j in range(self.region_num):
                if i == j:
                    self.m[i][j] = self.c_0
                else:
                    self.m[i][j] = (1 - self.c_0) * self.dist[i][j] / sum(self.dist[:,j])


if __name__ == '__main__':
    LP(2, [7378494, 3104614], 0, [0, 0], 2000, 2000 * 0.6, 0.5,
       [[0.07130173092733554, 0.06006747335430328], [0.06988767185595963, 0.0704837562356731],
        [0.05573469573621467, 0.02533586106459803], [0.23462177426447753, 0.16785599108408658],
        [0.00043765466861520264, 4.3614776105097235e-05], [4.897317495828119e-05, 0.0004750129321426092],
        [3.9324651150224985e-05, 0.00021986508113112352], [0.02085162747783373, 0.0041855223668634265],
        [0.0030803014321414976, 0.024376958306001326], [0.00919286669693171, 0.027962859103733615],
        0.39022872152097354, 0.7404707626634157, 0.9903998002421008],
       np.array([[0, 0.12095097], [0.12095097, 0]]),
       [7363861.03589267, 3094976.515156598],
       [1606.1338284020894, 820.666810038689],
       [5230.47417470184, 3192.8634192854433])
