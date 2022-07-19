"""
决策优化期末作业（实现两个非线性规划算法）

算法一：牛顿法（配合 外点法 实现分线性约束规划）

完成时间：2022-7-7

学生：@陈典，121351910001
"""
import numpy as np
from sympy import symbols, diff
from sympy.parsing.sympy_parser import parse_expr
import random


class Newton_Method:
    def __init__(self, function, constraints):
        """
        :param function: 输入的目标函数（形式如Word文档中示例）
        :param constraints: 输入的约束（形式如Word文档中示例）
        注：此为结构体中的 initial 函数，用于初始化自动运行，故没有返回值 return

        以下为自行设定的类的自身参数，用用于类内函数的调用（即：其他类内函数的输入参数）
        :param self.function: 类的自身参数，目标函数
        :param self.constraints: 类的自身参数，约束
        :param self.x: 类的自身参数，决策变量
        :param self.x_num: 类的自身参数，决策变量的数量
        :param self.epsilon: 类的自身参数，算法的要求精度
        :param self.epsilon_out: 类的自身参数，当问题有约束时，外循环的精度
        :param self.obj: 类的自身参数，最优函数值
        :param self.res: 类的自身参数，最优解
        """
        print(f'目标：{function}')
        print(f'约束：{constraints}')
        self.constraints = constraints
        if 'max' in function:  # 首先对输入的目标函数进行处理，让 sympy包 能够识别
            function = function.replace('max', '')
            function = function.replace('=', '')
            function = function.replace(' ', '')
            tag = 0
            function = list(function)
            for i in range(len(function)):
                if (i == 0) and (function[i] == '-'):  # 处理开头的负号
                    function[i] = ''
                elif (i == 0) and (function[i] != '-'):
                    tag = 1
                elif function[i] == '+':
                    function[i] = '-'
                elif function[i] == '-':
                    function[i] = '+'
            if tag == 1:  # 标记tag为1，则意味着开头需要添加负号，tag为0反之
                self.function = '-' + ''.join(function)
            else:
                self.function = ''.join(function)
        elif 'min' in function:
            self.function = function.replace('min', '')

        self.function = self.function.replace('=', '')
        self.function = self.function.replace(' ', '')
        self.function = self.function.replace('^', '**')
        self.function = self.function.replace('(', '')
        self.function = self.function.replace(')', '')
        # print('处理后',self.function)
        '--------------'
        self.x = []  # 初始化决策变量
        for i in range(5):  # 假定算例中最高可能存在五次项
            if f'x{i + 1}' in self.function:
                self.x.append(symbols(f'x{i + 1}'))
        self.x_num = len(self.x)  # 变量数量
        # print(self.x)
        '--------------'
        self.epsilon = 1.0e-03  # 算法的要求精度
        self.epsilon_out = 1.0e-03  # 当问题有约束时，外循环的精度
        self.obj = 0
        self.res = []
        '--------------'
        if not constraints:
            self.obj, self.res = self.algorithm_no_cons()  # 运行无约束算法
        else:
            self.obj, self.res = self.algorithm_cons()  # 运行有约束算法

        for i in range(self.x_num):
            self.res[i] = round(self.res[i], 3)
        self.obj = round(self.obj, 3)
        print(f'最优解为：{self.res}（精确到3位小数）')
        print(f'最优函数值为：{self.obj}（精确到3位小数）')
        print(f'___________________________________')

    def algorithm_no_cons(self):  # 函数：无约束的算法入口
        """
        以下为该函数调用的类的自身参数，按照作业要求，可以看做函数的输入
        :param self.function: 类的自身参数，目标函数
        :param self.x: 类的自身参数，决策变量
        :param self.x_num: 类的自身参数，决策变量的数量
        :param self.epsilon: 类的自身参数，算法的要求精度

        :return obj: 算法得到的最优目标函数值
        :return x: 算法得到的最优解
        """
        # （1）给定初始解
        x_initial = np.array([random.uniform(-2, 2) for i in range(len(self.x))], dtype=float)
        x = x_initial

        # （2-1）求一阶导数
        Gradient = self.gradient(x, self.function)
        # print(f'Gradient:{Gradient}')

        # （2-2）求海森矩阵
        Hessian = self.hessian(x, self.function)
        # print(f'Hessian:{Hessian}')

        # （3）判断精度并循环
        iteration = 0
        while True:
            if np.linalg.norm(Gradient) < self.epsilon:  # 判断是否达到要求的精度，若达到即可退出迭代
                break
            iteration += 1
            # （4）计算下一个迭代点
            print(f'第{iteration}次迭代，当前解为：{x}')
            Hessian_T = np.linalg.inv(Hessian)
            calculate = np.matmul(Hessian_T, Gradient)

            x = x - calculate * 1  # 步长默认取1

            Gradient = self.gradient(x, self.function)  # 更新一、二阶导数
            Hessian = self.hessian(x, self.function)

        print(f'满足精度：{self.epsilon}，停止迭代')

        obj = parse_expr(self.function)  # 把字符串形式转换为表达式形式，计算目标函数值
        for i in range(self.x_num):
            obj = obj.evalf(subs={self.x[i]: x[i]})  # 依次代入

        return obj, x

    def gradient(self, x_initial, function):  # 函数：求一阶导数
        """
        :param x_initial: 输入的解
        :param function: 输入的目标函数

        以下为该函数调用的类的自身参数，按照作业要求，可以看做函数的输入
        :param self.x: 类的自身参数，决策变量
        :param self.x_num: 类的自身参数，决策变量的数量

        :return Gradient: 函数计算得到的一阶导数矩阵
        """
        Gradient = np.array([None for i in range(self.x_num)])  # 初始化矩阵

        for i in range(self.x_num):  # 求导
            Gradient[i] = diff(function, self.x[i])  # 形成表达式形式
            for j in range(self.x_num):
                Gradient[i] = Gradient[i].subs(self.x[j], x_initial[j])  # 依次代入初始解
        Gradient = Gradient.astype(float)  # 矩阵元素全部转换为float

        return Gradient

    def hessian(self, x_initial, function):  # 函数：求海森矩阵
        """
        :param x_initial: 输入的解
        :param function: 输入的目标函数

        以下为该函数调用的类的自身参数，按照作业要求，可以看做函数的输入
        :param self.x: 类的自身参数，决策变量
        :param self.x_num: 类的自身参数，决策变量的数量

        :return Hessian: 函数计算得到的海森矩阵
        """
        Hessian = np.array([[None for i in range(self.x_num)] for j in range(self.x_num)])  # 初始化矩阵

        for i in range(self.x_num):
            for j in range(self.x_num):  # 求二阶导
                Hessian[i][j] = diff(diff(function, self.x[i]), self.x[j])  # 形成表达式形式
                for k in range(self.x_num):
                    Hessian[i][j] = Hessian[i][j].subs(self.x[k], x_initial[k])  # 依次代入初始解
        Hessian = Hessian.astype(float)  # 矩阵元素全部转换为float

        return Hessian

    def algorithm_cons(self):  # 函数：有约束的算法入口
        """
        以下为该函数调用的类的自身参数，按照作业要求，可以看做函数的输入
        :param self.function: 类的自身参数，目标函数
        :param self.constraints: 类的自身参数，约束
        :param self.x: 类的自身参数，决策变量
        :param self.x_num: 类的自身参数，决策变量的数量
        :param self.epsilon: 类的自身参数，算法的要求精度
        :param self.epsilon_out: 类的自身参数，当问题有约束时，外循环的精度

        :return obj: 算法得到的最优目标函数值
        :return x: 算法得到的最优解
        """
        unequal = []  # 用于存放整理后的约束表达式（不等式约束）
        equal = []  # 用于存放整理后的约束表达式（等式约束）
        for st in self.constraints:  # 对约束条件进行整理，让 sympy包 能够识别
            st = st.replace('(', '')
            st = st.replace(')', '')
            st = st.replace('^', '**')
            st = st.replace(' ', '')
            if '>' in st:
                st = st.replace('>', '-')
                unequal.append(st)
            elif '<' in st:
                st = list(st)
                tag = 0
                insert = -1
                for i in range(len(st)):
                    if (i == 0) and (st[i] == '-'):  # 处理开头的负号
                        st[i] = ''
                    elif (i == 0) and (st[i] != '-'):
                        tag = 1
                    elif st[i] == '<' and st[i + 1] != '-':
                        insert = i
                    elif st[i] == '+':
                        st[i] = '-'
                    elif st[i] == '-':
                        st[i] = '+'
                if insert != -1:
                    st.insert(insert, '-')
                if tag == 1:
                    st = '-' + ''.join(st)
                else:
                    st = ''.join(st)
                st = st.replace('<', '')
                unequal.append(st)
            elif '=' in st:
                st = st.replace('=', '-')
                equal.append(st)
        # print(equal, unequal)

        # （1）给定初始解
        x_initial = np.array([random.uniform(-2, 2) for i in range(len(self.x))], dtype=float)
        x = x_initial
        penalty = 1  # 惩罚因子
        alpha = 1.5  # 惩罚因子的放大系数

        # 生成无约束问题，内外两循环，求解
        iteration = 0  # 判断精度并循环
        while True:  # 外循环
            iteration += 1
            print(f'第{iteration}次外循环，当前解为：{x}')

            function = parse_expr(self.function)  # 将约束变成惩罚项，加到目标函数中去
            for eq in equal:  # 等式约束直接平方加上去
                function = function + penalty * parse_expr(eq) ** 2
            for ueq in unequal:  # 不等式约束需要判断是否满足，选择性地加上去
                tem = parse_expr(ueq)
                for i in range(self.x_num):
                    tem = tem.evalf(subs={self.x[i]: x[i]})  # 依次代入（计算约束值）
                if tem < 0:
                    function = function + penalty * parse_expr(ueq) ** 2
            # print(function)

            Gradient = self.gradient(x, function)  # 求一阶导数
            # print(f'Gradient:{Gradient}')

            Hessian = self.hessian(x, function)  # 求海森矩阵
            # print(f'Hessian:{Hessian}')

            # （2）内循环开始，同理无约束的算法，采取牛顿法（故不进行详细的注释了）
            while True:
                if np.linalg.norm(Gradient) < self.epsilon:
                    break
                Hessian_T = np.linalg.inv(Hessian)
                calculate = np.matmul(Hessian_T, Gradient)

                x = x - calculate * 1

                Gradient = self.gradient(x, function)
                Hessian = self.hessian(x, function)

            print(f'满足精度：{self.epsilon}，停止迭代')

            # （3）判断是否满足约束条件，即：外循环的退出条件，1满足精度，可以退出，0反之
            break_tag = 1
            for eq in equal:
                tem = parse_expr(eq)
                for i in range(self.x_num):
                    tem = tem.evalf(subs={self.x[i]: x[i]})  # 依次代入（计算约束值）
                if abs(tem) >= self.epsilon_out:
                    break_tag = 0

            for ueq in unequal:
                tem = parse_expr(ueq)
                for i in range(self.x_num):
                    tem = tem.evalf(subs={self.x[i]: x[i]})  # 依次代入（计算约束值）
                if round(tem, 3) < 0:
                    break_tag = 0

            # （4）停止循环，或对惩罚因子进行放大并继续循环
            if break_tag == 1:
                break
            else:
                penalty = penalty * alpha  # 放大惩罚因子
                continue

        obj = parse_expr(self.function)  # 把字符串形式转换为表达式形式，计算最优目标函数值
        for i in range(self.x_num):
            obj = obj.evalf(subs={self.x[i]: x[i]})  # 依次代入

        return obj, x


def repeat_experiment(f, c):  # 函数：重复试验
    """
    :param f: 目标函数
    :param c: 约束

    :return best_obj: 最优目标函数值
    :return best_res: 最优解
    """
    res = []
    obj = []
    best_res = []
    best_obj = float('inf')
    for i in range(5):
        sol = Newton_Method(f,c)
        res.append(sol.res)
        obj.append(sol.obj)
        if sol.obj < best_obj:
            best_obj = sol.obj
            best_res = sol.res
    print(f'为了考虑初始点对局部最优解的影响，本程序将随机生成5个初始点，进行计算，并取出其最优解')
    for i in range(5):
        print(f'第{i + 1}组：res = {res[i]}, obj = {obj[i]}')

    return best_obj, best_res


if __name__ == '__main__':
    # 示例：
    # func = 'min = x(1)^2 - 4 * x(1) + x(2)^2 - 2*x(2) + 5'  # 2,1
    # cons = []

    func = 'min = x(1) ^ 2 + x(2) ^ 2 + 8'  # 1,1
    # 约束条件List[String]（>表示大于等于，<表示小于等于）
    cons = ['x(2) - x(1) ^ 2 < 0', 'x(1) + x(2) ^ 2 = 2', 'x(1) > 0', 'x(2) > 0']

    # 因为重复试验，所以程序运行起来需要稍等一段运算时间~
    obj_best, res_best = repeat_experiment(func, cons)  # 执行算法
    print(f'经过五组初始点求解结果对比，最优解为：res = {res_best}, obj = {obj_best}（精确到3位小数）')
    print(f'___________________________________')
    print(f'注：由于精度问题，直观的结果可能会在小数点后很多位处，有极小的差异，例如：10显示为9.99999等')
    print(f'注：程序默认输入的表达式最高为5次方，但是可以灵活变动，修改 line 55 处的数字即可')
    print(f'注：程序默认输入的惩罚因子初始值为1，惩罚因子放大系数为1.5，但是可以灵活变动，修改 line 232/233 处的数字即可')
    print(f'___________________________________')
