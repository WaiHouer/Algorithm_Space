"""
作业第一题：分支定界法

完成时间：2022-5-15

author：@陈典
"""
from scipy.optimize import linprog
import math


class Branch_and_Bound:
    """
    分支定界法
    注意：min问题（约束形式为 Ax <= b）。
    """
    def __init__(self,c,a_ub,b_ub,a_eq,b_eq,bounds):
        self.upper_bound = 999999999  # 初始化问题上界
        self.final_obj = None  # 最终目标函数值
        self.final_sol = None  # 最终解

        # 我采用堆栈形式进行分支
        # 较差的分支先压入，较好的分支后压入，每次取出一个分支进行分支定界，如此往复
        # 这样，可以做到差的留在底部（Best策略），好的先进行深度（Depth策略）分支定界
        self.stack = []
        # 注：若要best+广度，则需要用队列，且好的分支先进队，差的分支后进队！！

        self.c = c        # 目标函数系数
        self.a_ub = a_ub  # 不等式系数
        self.b_ub = b_ub  # 不等式右侧值
        self.a_eq = a_eq  # 等式系数
        self.b_eq = b_eq  # 等式右侧值
        self.bounds = bounds  # 变量取值区间组合

        print('特别说明：')
        print('由于程序计算精度问题，有时候0会计算成一个极小但是非0的数，会多进行一次无效的分支')
        print('并且，我是用堆栈来模拟分支过程的，属于是深度优先')
        print('因此，分支定界次数可能有一点多，但是完全不影响最终结果！')
        print('-------------------------')
        print('第一题，开始：')
        print('-------------------------')

        sol = linprog(self.c,self.a_ub,self.b_ub,self.a_eq,self.b_eq,self.bounds)  # 正常求一次

        if not sol.success:
            print('该问题无解!!')
        else:
            self.stack.append([sol,self.a_ub,self.b_ub])  # 否则，将初始解、不等式系数、不等式右侧  打包压入堆栈

            self.algorithm()

        self.final_set = [self.final_sol, self.final_obj]  # 打包，方便调用返回值

        print(f'所有分支修剪完毕：')
        print(f'作业第一题，最优解：{self.final_sol}')
        print(f'作业第一题，最优函数值：{self.final_obj}')

    def algorithm(self):
        iteration = 0
        while self.stack:  # 堆栈不空，循环
            iteration += 1

            sol,a_ub,b_ub = self.stack.pop()  # 取出一个节点
            print(f'当前迭代次数{iteration}次，该节点最优解，以及最优函数值：')
            print(f'最优解：{sol.x}，最优函数值：{sol.fun}')

            if sol.fun > self.upper_bound:  # 如果比上界差，剪掉该分支
                print(f'目标函数值比上界差，剪掉该分支')
                print('-------------------------')
                continue

            if all(list(map(lambda f: f.is_integer(),sol.x))):  # 如果变量刚好全都是整数了已经，则更新
                if sol.fun < self.upper_bound:
                    self.upper_bound = sol.fun
                    self.final_obj = sol.fun
                    self.final_sol = sol.x
                    print(f'找到一组整数解，更新上界：{self.upper_bound}')
                    print('-------------------------')
                continue

            else:  # 否则，分支定界
                gap = 11   # 初始化两侧差值
                index = -1  # 初始化选中的下标
                for i in range(len(sol.x)):  # 选择出：离两侧整数较远的分数，并记录其下标位置
                    if not sol.x[i].is_integer():
                        current_gap = math.ceil(sol.x[i]) - math.floor(sol.x[i])  # 差值越小，两侧距离越远
                        if current_gap < gap:
                            index = i
                print(f'选择变量x[{index + 1}]进行分支')

                # >=分支
                a_ub_up = []  # 添加的不等式
                for i in range(len(a_ub[0])):
                    if i == index:
                        a_ub_up.append(-1)  # >=取负数，变成<=不等式
                    else:
                        a_ub_up.append(0)  # 其他变量在该不等式中不存在
                b_ub_up = -math.ceil(sol.x[index])  # 添加的不等式右侧值
                print(f'不等式分支一：x[{index + 1}] >= {-b_ub_up}')

                # >=分支
                a_ub_down = []
                for i in range(len(a_ub[0])):
                    if i == index:
                        a_ub_down.append(1)
                    else:
                        a_ub_down.append(0)
                b_ub_down = math.floor(sol.x[index])
                print(f'不等式分支二：x[{index + 1}] <= {b_ub_down}')
                print('-------------------------')

                a_ub_1 = a_ub + [a_ub_up]
                b_ub_1 = b_ub + [b_ub_up]
                sol_1 = linprog(self.c,a_ub_1,b_ub_1,self.a_eq,self.b_eq,self.bounds)  # 计算>=分支

                a_ub_2 = a_ub + [a_ub_down]
                b_ub_2 = b_ub + [b_ub_down]
                sol_2 = linprog(self.c,a_ub_2,b_ub_2,self.a_eq,self.b_eq,self.bounds)  # 计算<=分支

                # 有效的、较差的先压入堆栈，有效的、较好的后压入堆栈
                if sol_1.success and not sol_2.success:
                    self.stack.append([sol_1,a_ub_1,b_ub_1])
                elif sol_2.success and not sol_1.success:
                    self.stack.append([sol_2,a_ub_2,b_ub_2])
                elif sol_1.success and sol_2.success:
                    if sol_1.fun < sol_2.fun:
                        self.stack.append([sol_2,a_ub_2,b_ub_2])
                        self.stack.append([sol_1,a_ub_1,b_ub_1])
                    else:
                        self.stack.append([sol_1,a_ub_1,b_ub_1])
                        self.stack.append([sol_2,a_ub_2,b_ub_2])


if __name__ == '__main__':
    # 输入目标函数系数（min问题）
    c1 = [4, 7, 3]
    # 输入s.t.不等式系数（均为<=，按行输入）
    A_ub1 = [[-1, -3, -1],[-3, -1, -2]]
    # 输入不等式右侧
    b_ub1 = [-5, -8]
    # 输入s.t.等式系数（按行输入）
    A_eq1 = None
    # 输入等式右侧
    b_eq1 = None
    # 输入变量约束区间
    x1_bounds = [0,None]
    x2_bounds = [0,None]
    x3_bounds = [0,None]
    bounds1 = (x1_bounds,x2_bounds,x3_bounds)
    Branch_and_Bound(c1,A_ub1,b_ub1,A_eq1,b_eq1,bounds1)
