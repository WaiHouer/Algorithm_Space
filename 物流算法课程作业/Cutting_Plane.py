"""
作业第二题：割平面法

完成时间：2022-5-16

author：@陈典
"""
from ortools.linear_solver import pywraplp
import numpy as np


class CutPlane:
    """
    割平面法
    注意：max问题（约束形式为 Ax <= b）。
    """
    def __init__(self, c, A, b, lb=None, ub=None):
        # 输入
        self.c = np.array(c) * 1.0
        self.A = np.array(A) * 1.0
        self.b = np.array(b) * 1.0
        self.lb = lb
        self.ub = ub
        # 输出
        self._sol = None  # solution
        self._obj_val = None  # objective value
        # 辅助变量
        self._iter_num = 0
        self._obj = None
        self.c1 = None
        self.A1 = None
        self.b1 = None
        self._model = None
        self._x1 = []
        self._sol1 = None
        self.const1 = []
        self.basic_consts = None
        self.basic_vars = None
        self.non_basic_vars = None
        self.basic_matrix = None
        self.non_basic_matrix = None

        print('特别说明：')
        print('由于需要对最终表做一些变化，所以我选择用了ortools这个包求解线性规划')
        print('因此，和分支定界用的包不一样，分支定界只是为了语句简单')
        print('-------------------------')
        print('第二题，计算开始：')
        print('-------------------------')

    def init_lb_ub(self):
        # 初始化上下界
        n = len(self.c)
        if self.lb is None:
            self.lb = [None] * n
        if self.ub is None:
            self.ub = [None] * n

    def init_model(self):
        self._model = pywraplp.Solver.CreateSolver('GLOP')  # 初始化模型
        # 约束：Ax <= b
        m, n = np.shape(self.A)
        self._x1 = [self._model.NumVar(0, self._model.Infinity(), 'x[%d]' % j) for j in range(n)]
        for i in range(m):
            self.add_cut(self._x1, self.A[i, :], self.b[i])
        # 约束：lb <= x <= ub
        self.init_lb_ub()
        for j in range(n):
            # x <= ub
            if self.ub[j] is not None:
                self.add_cut([self._x1[j]], [1], self.ub[j])
            # -x <= -lb
            if self.lb[j] is not None:
                self.add_cut([self._x1[j]], [-1], -self.lb[j])
        # 目标
        self._obj = self._model.Objective()
        self.c1 = self.c
        for j in range(n):
            self._obj.SetCoefficient(self._x1[j], self.c1[j])
        self._obj.SetMaximization()

    def add_cut(self, variables, coefficients, ub):  # 添加割平面约束
        # 把不等式约束写成等式约束
        ct = self._model.Constraint(ub, ub)
        for x, c in zip(variables, coefficients):
            ct.SetCoefficient(x, c)
        # 松弛约束
        slack_var = self._model.NumVar(0, self._model.Infinity(), 'x[%d]' % len(self._x1))
        ct.SetCoefficient(slack_var, 1)
        self._x1.append(slack_var)
        self.const1.append(ct)

    def standard(self):  # 把问题用标准形表示：min c1 * x1 s.t. A1 * x1 = b1
        m, n = len(self.const1), len(self._x1)
        self.c1 = np.array(self.c.tolist() + [0] * (n - len(self.c1)))
        self.A1 = np.zeros((m, n))
        self.b1 = np.zeros(m)
        for i in range(m):
            for j in range(n):
                self.A1[i][j] = self.const1[i].GetCoefficient(self._x1[j])
            self.b1[i] = self.const1[i].Ub()

    def add_cuts_batch(self):  # 计算并添加割平面
        B_inv = np.linalg.inv(self.basic_matrix)
        N_bar = B_inv @ self.non_basic_matrix
        b1 = [self.b1[i] for i in self.basic_consts]
        b_bar = B_inv @ b1
        # 生成
        variables = [self._x1[j] for j in self.non_basic_vars]
        coefficients_batch = np.array([-N_bar[:, j] + np.floor(N_bar[:, j])
                                       for j in range(len(self.non_basic_vars))]).T
        ub_batch = -b_bar + np.floor(b_bar)
        # 添加
        for coefficients, ub in zip(coefficients_batch, ub_batch):
            self.add_cut(variables, coefficients, ub)

    def solve_lp(self):  # 求解松弛问题
        self._model.Solve()
        # 把问题转换成标准化形式
        self.standard()
        # 然后计算辅助变量
        # 为下一步计算割平面做准备
        m, n = np.shape(self.A1)
        self._sol1 = [self._x1[j].solution_value() for j in range(n)]
        self._sol = self._sol1[0: len(self.c)]  # 原问题的解
        self._obj_val = self._obj.Value()  # 目标函数值
        self.basic_vars = [j for j in range(n) if self._x1[j].basis_status() == self._model.BASIC]
        self.non_basic_vars = list(set(range(n)) - set(self.basic_vars))
        self.basic_consts = np.array([j for j in range(m) if self.const1[j].basis_status() == self._model.FIXED_VALUE])
        self.basic_matrix = np.array([[self.A1[i][j]
                                       for j in self.basic_vars] for i in self.basic_consts])
        self.non_basic_matrix = np.array([[self.A1[i][j]
                                           for j in self.non_basic_vars] for i in self.basic_consts])
        print(f'当前基矩阵：\n{self.basic_matrix}，\n当前非基矩阵：\n{self.non_basic_matrix}')

    def is_feasible(self):
        for v in self._sol1:
            if abs(v - np.round(v)) > 1e-6:  # 计算精度问题
                return False
        return True

    def solve(self):
        self.init_model()  # 初始化松弛问题
        print(f'迭代次数：{self._iter_num + 1}')
        self.solve_lp()  # 求解松弛问题
        print(f'最优解：{np.round(self._sol, 5)}')
        print(f'最优函数值：{- np.round(self._obj_val, 5)}')
        print('-------------------------')
        while not self.is_feasible():
            self._iter_num += 1
            self.add_cuts_batch()  # 添加割平面
            print(f'迭代次数：{self._iter_num + 1}')
            self.solve_lp()  # 求解
            print(f'最优解：{np.round(self._sol, 5)}')
            print(f'最优函数值：{- np.round(self._obj_val, 5)}')
            print('-------------------------')

        print(f'此时，完成迭代，得到整数解：')
        print(f'最优解：{np.round(self._sol, 5)}')
        print(f'最优函数值：{- np.round(self._obj_val, 5)}')


if __name__ == '__main__':
    c_ = [-5, -3]
    A_ = [[-2, -1], [-1, -3]]
    b_ = [-10, -9]
    lb_ = None
    ub_ = None
    CutPlane(c_, A_, b_, lb_, ub_).solve()
