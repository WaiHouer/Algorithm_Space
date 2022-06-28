"""
作业第四题：Benders分解法

完成时间：2022-5-19

author：@陈典
"""
from gurobipy import *


class Benders_Decomposition:
    def __init__(self):
        self.MP = Model()  # Benders分解的主问题（Master Problem）
        self.SP_Dual = Model()  # Benders分解的子问题的对偶（Dual of SubProblem）
        self.MP.setParam('Outputflag', 0)
        self.SP_Dual.setParam('Outputflag', 0)

        # 添加决策变量（y 和 q）
        self.y = self.MP.addVars(5, obj=7, vtype=GRB.BINARY, name='y')
        self.q = self.MP.addVar(obj=1, vtype=GRB.CONTINUOUS, name='q')
        # 添加对偶问题变量
        self.dual_1 = self.SP_Dual.addVars(3, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='dual_1')
        self.dual_2 = self.SP_Dual.addVars(5, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name='dual_2')

        # 添加对偶问题约束
        self.constraint_1 = self.SP_Dual.addConstr(self.dual_1[0] + self.dual_2[0] <= 1, name='constraint_1')
        self.constraint_2 = self.SP_Dual.addConstr(self.dual_1[1] + self.dual_2[1] <= 1, name='constraint_2')
        self.constraint_3 = self.SP_Dual.addConstr(self.dual_1[2] + self.dual_2[2] <= 1, name='constraint_3')
        self.constraint_4 = self.SP_Dual.addConstr(self.dual_1[0] + self.dual_1[2] + self.dual_2[3] <= 1,
                                                   name='constraint_4')
        self.constraint_5 = self.SP_Dual.addConstr(self.dual_1[0] + self.dual_1[1] + self.dual_2[4] <= 1,
                                                   name='constraint_5')

        # 设置参数 InfUnbdInfo
        self.SP_Dual.Params.InfUnbdInfo = 1

        self.iteration = 0
        self.SP_Dual_obj = [9999]  # 设置一个足够大的数

        print('-------------------------')
        print('第四题，开始：')
        print('-------------------------')

        self.algorithm()

    def algorithm(self):
        x = []
        self.MP.optimize()
        print(f'第1次计算，松弛主问题（Relax Master Problem）：')
        y_q = self.MP.getAttr('x')
        y_abs = list(map(abs, y_q[0:-1]))  # 对-0取个绝对值，好看一些
        print(f'y_hat(1)：{y_abs}')
        print(f'q(1)：{y_q[-1]}')  # 最后一个是q
        print(f'将两者传入子问题的对偶问题（Dual of SubProblem）')
        print('-------------------------')


        while self.SP_Dual_obj[0] > self.q.x:
            if self.iteration == 0:
                self.SP_Dual.setObjective(8 * self.dual_1[0] + 3 * self.dual_1[1] + 5 * self.dual_1[2] +
                                          8 * self.dual_2[0] * self.y[0].x + 3 * self.dual_2[1] * self.y[1].x +
                                          5 * self.dual_2[2] * self.y[2].x + 5 * self.dual_2[3] * self.y[3].x +
                                          3 * self.dual_2[4] * self.y[4].x, GRB.MAXIMIZE)

                self.SP_Dual.optimize()
                self.add_benders_cut(x)  # 添加Benders割
                self.iteration = 1

            else:
                self.dual_2[0].obj = 8 * self.y[0].x
                self.dual_2[1].obj = 3 * self.y[1].x
                self.dual_2[2].obj = 5 * self.y[2].x
                self.dual_2[3].obj = 5 * self.y[3].x
                self.dual_2[4].obj = 3 * self.y[4].x

                self.SP_Dual.optimize()
                self.add_benders_cut(x)  # 添加Benders割
                self.iteration += 1

            self.MP.optimize()
            print(f'第{self.iteration + 1}次计算，松弛主问题（Relax Master Problem）：')
            y_q = self.MP.getAttr('x')
            y_abs = list(map(abs, y_q[0:-1]))  # 对-0取个绝对值，好看一些
            print(f'y_hat({self.iteration + 1})：{y_abs}')
            print(f'q({self.iteration + 1})：{y_q[-1]}')  # 最后一个是q
            if self.SP_Dual_obj[0] > self.q.x:
                print(f'将两者传入子问题的对偶问题（Dual of SubProblem）')
            else:
                print(f'达到终止条件，计算完毕')
            print('-------------------------')

        print('最终计算得到最优解：')
        print(f'x = {x}')
        y = []
        for i in range(5):
            y.append(abs(self.y[i].x))
        print(f'y = {y}')
        print(f'最优函数值：opt = {sum(x) + 7 * sum(y)}')

    def add_benders_cut(self, x):
        if self.SP_Dual.status == GRB.Status.UNBOUNDED:  # 如果子问题对偶是无界的
            d = self.SP_Dual.UnbdRay  # 获取极方向
            print(f'第{self.iteration + 1}次计算，子问题的对偶问题（Dual of SubProblem）：')
            print(f'无界解，极方向 d{self.iteration + 1}：{d}')
            print(f'以此生成 Feasibility_Cut，传回松弛主问题（Relax Master Problem）')
            print('-------------------------')
            self.MP.addConstr(8 * d[0] + 3 * d[1] + 5 * d[2] +
                              8 * d[3] * self.y[0] + 3 * d[4] * self.y[1] + 5 * d[5] * self.y[2] +
                              5 * d[6] * self.y[3] + 3 * d[7] * self.y[4] <= 0, name='Feasibility_Cut')

        elif self.SP_Dual.status == GRB.Status.OPTIMAL:  # 发现最优解
            print(f'第{self.iteration + 1}次计算，子问题的对偶问题（Dual of SubProblem）：')
            print(f'有最优解，极点 w：{self.SP_Dual.x}')
            print(f'以此生成 Optimality_Cut，传回松弛主问题（Relax Master Problem）')
            print('-------------------------')
            self.MP.addConstr(8 * self.dual_1[0].x + 3 * self.dual_1[1].x + 5 * self.dual_1[2].x +
                              8 * self.dual_2[0].x * self.y[0] + 3 * self.dual_2[1].x * self.y[1] +
                              5 * self.dual_2[2].x * self.y[2] + 5 * self.dual_2[3].x * self.y[3] +
                              3 * self.dual_2[4].x * self.y[4] <= self.q, name='Optimality_Cut')
            self.SP_Dual_obj[0] = self.SP_Dual.ObjVal  # 存储最优值
            x.append(self.constraint_1.pi)
            x.append(self.constraint_2.pi)
            x.append(self.constraint_3.pi)
            x.append(self.constraint_4.pi)
            x.append(self.constraint_5.pi)
        else:
            print(self.SP_Dual.status)


if __name__ == '__main__':
    Benders_Decomposition()
