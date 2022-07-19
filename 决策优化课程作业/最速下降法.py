"""
决策优化 - 最速下降法作业实现
姓名：詹妍
学号：121351910009
"""
from sympy import symbols, diff
import numpy as np
from sympy.parsing.sympy_parser import parse_expr


def function_deal(function, tag):  # 函数：对输入进行标准化处理
    function = function.replace('=', '')
    function = function.replace('(', '')
    function = function.replace(')', '')
    function = function.replace('^', '**')
    if tag == 'min':
        function = function.replace('min', '')
    elif tag == 'max':
        function = function.replace('max', '')  # 如果是最大化问题，全都取负数
        negative = 0
        function = list(function)
        for i in range(len(function)):
            if (i == 0) and (function[i] == '-'): function[i] = ''
            elif (i == 0) and (function[i] != '-'): negative = 1
            elif function[i] == '+': function[i] = '-'
            elif function[i] == '-': function[i] = '+'
        if negative == 1:
            function = '-' + ''.join(function)
        else:
            function = ''.join(function)
    return function


def g_calculate(x, function, number, x_symbol):
    g = np.array([None for i in range(number)])
    for i in range(number):
        g[i] = diff(function, x_symbol[i])
        for kk in range(number):
            g[i] = g[i].subs(x_symbol[kk], x[kk])
    return g


def g_calculate_2(x, function, number, x_symbol):
    g_2 = np.array([[None for m in range(number)] for n in range(number)])
    for i in range(number):
        for j in range(number):
            g_2[i][j] = diff(diff(function, x_symbol[i]), x_symbol[j])
            for kk in range(number):
                g_2[i][j] = g_2[i][j].subs(x_symbol[kk], x[kk])
    return g_2


def decent_1(function, number, x_symbol, jing_du_1):  # 函数：牛顿法——无约束情况
    x = [0 for i in range(number)]

    g = g_calculate(x, function, number, x_symbol).astype(float)  # 导数

    n = 0
    while True:
        n += 1
        print(f'第{n}次的x：{x}')

        q = symbols('q')
        x_1 = x - g * q
        f_tem = parse_expr(function)
        for i in range(number):
            f_tem = f_tem.evalf(subs={x_symbol[i]: x_1[i]})
        q_d = golden(f_tem, q)
        x = x - g * q_d

        g = g_calculate(x, function, number, x_symbol).astype(float)  # 更新

        norm_2 = np.linalg.norm(g)
        if norm_2 < jing_du_1:  # 判断是否达到要求的精度，若达到即可退出迭代
            break

    tem = parse_expr(function)
    for i in range(number):
        tem = tem.evalf(subs={x_symbol[i]: x[i]})
    return tem, x


def golden(function, q):
    low, up = -50, 50
    m = low + 0.382 * (up - low)
    n = low + 0.618 * (up - low)

    while True:
        if up - low < 0.01:
            min_p = (low + up) / 2
            break
        f_1 = function.evalf(subs={q: m})
        f_2 = function.evalf(subs={q: n})
        if f_1 > f_2: low = m; m = n; n = low + 0.618 * (up - low)
        else: up = n; n = m; m = low + 0.382 * (up - low)
    return min_p


def decent_2(function_tem, number, x_symbol, jing_du_1, jing_du_2, st_1, st_2):  # 函数：牛顿法——有约束情况
    x = [0 for i in range(number)]
    a_1 = 1  # 惩罚系数
    a_2 = 1.5  # 放大倍数

    n = 0
    while True:  # 两层循环，外层是添加惩罚项，内层是牛顿法
        n += 1
        print(f'第{n}次循环的x：{x}')
        function = parse_expr(function_tem)
        for s in st_1:
            ss = parse_expr(s)
            for i in range(number):
                ss = ss.evalf(subs={x_symbol[i]: x[i]})
            if ss < 0:
                function = function + a_1 * parse_expr(s) ** 2  # 不等式惩罚项
        for s in st_2:
            function = function + a_1 * parse_expr(s) ** 2  # 等式添加惩罚项

        g = g_calculate(x, function, number, x_symbol).astype(float)  # 导数
        while True:
            q = symbols('q')
            x_1 = x - g * q
            f_tem = function
            for i in range(number):
                f_tem = f_tem.evalf(subs={x_symbol[i]: x_1[i]})
            q_d = golden(f_tem, q)
            x = x - g * q_d
            g = g_calculate(x, function, number, x_symbol).astype(float)  # 更新
            norm_2 = np.linalg.norm(g)
            if norm_2 < jing_du_1:  # 判断是否达到要求的精度，若达到即可退出迭代
                break

        e_tag = 1  # 约束是否满足，满足则结束计算
        for s in st_2:
            ss = parse_expr(s)
            for i in range(number):
                ss = ss.evalf(subs={x_symbol[i]: x[i]})
            if abs(ss) >= jing_du_2:
                e_tag = 0
        for s in st_1:
            ss = parse_expr(s)
            for i in range(number):
                ss = ss.evalf(subs={x_symbol[i]: x[i]})
            if round(ss, 2) < 0:
                e_tag = 0
        if e_tag == 1:
            break
        else:
            a_1 = a_1 * a_2  # 更新惩罚系数

    tem = parse_expr(function_tem)
    for i in range(number):
        tem = tem.evalf(subs={x_symbol[i]: x[i]})
    return tem, x


def main_f(function, sts):  # 主函数
    function = function.replace(' ', '')  # 若有空格，去掉
    if 'max' in function:
        function = function_deal(function, 'max')  # 对目标函数表达式进行处理
    elif 'min' in function:
        function = function_deal(function, 'min')
    st_1, st_2 = [], []  # 不等式和等式约束
    if sts:
        for s in sts:
            s = s.replace(' ', '')
            s = s.replace('^', '**')
            s = s.replace('(', '')
            s = s.replace(')', '')
            if '>' in s: s = s.replace('>', '-'); st_1.append(s)
            elif '=' in s: s = s.replace('=', '-'); st_2.append(s)
            elif '<' in s:
                s, negative, ins = list(s), 0, -1
                for i in range(len(s)):
                    if (i == 0) and (s[i] == '-'): s[i] = ''
                    elif (i == 0) and (s[i] != '-'): negative = 1
                    elif s[i] == '<' and s[i + 1] != '-': ins = i
                    elif s[i] == '+': s[i] = '-'
                    elif s[i] == '-': s[i] = '+'
                if ins >= 0: s.insert(ins, '-')
                if negative == 1: s = '-' + ''.join(s)
                else: s = ''.join(s)
                s = s.replace('<', '')
                st_1.append(s)

    number = 0
    for i in range(3):
        if f'x{i+1}' in function:
            number += 1
    x_symbol = [symbols(f'x{i+1}') for i in range(number)]  # 决策变量
    jing_du_1, jing_du_2 = 0.01, 0.01  # 精度

    if not sts:
        obj, res = decent_1(function, number, x_symbol, jing_du_1)  # 无约束
    else:
        obj, res = decent_2(function, number, x_symbol, jing_du_1, jing_du_2, st_1, st_2)

    return res, obj


if __name__ == '__main__':
    func = 'min = x(1)^2 - 4 * x(1) + x(2)^2 - 2*x(2) + 5'  # 2,1
    cons = []

    best_res, best_obj = main_f(func, cons)
    print('最终结果：')
    print(f'最优解为：{best_res}，最优函数值为：{best_obj}')
