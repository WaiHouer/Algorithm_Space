"""
短视迭代优化算法——Myopic LP
之：当期整数规划模型求解
"""
import numpy as np
import gurobipy as gp
from numba import njit


def myopic_model(K, S, E, A, U, b_last, N, sigma_hat, beta_e, beta_a, beta_u
                 , eta, b_hat, lambda_b, C, lambda_c):
    model = gp.Model()  # 建立模型

    b = model.addVars(K, lb=0, vtype=gp.GRB.INTEGER)  # 规定变量（正整数）
    c = model.addVars(K, lb=0, vtype=gp.GRB.INTEGER)
    model.update()  # 更新变量空间

    constant_e, constant_a, constant_u, coefficient_b, coefficient_c = coefficient_calculate(K, S, E, A, U, N, sigma_hat
                                                                                             , beta_e, beta_a, beta_u
                                                                                             , eta)
    model.setObjective(constant_e + constant_a + constant_u
                       - gp.quicksum(b[j] * coefficient_b[j] for j in range(K))
                       + gp.quicksum(c[j] * coefficient_c[j] for j in range(K))
                       , gp.GRB.MINIMIZE)  # 规定目标函数

    B = sum(b_hat)  # 约束1（其实不算是约束）
    model.addConstr(b.sum('*') <= B)
    model.addConstrs(b[k] >= b_last[k] for k in range(K))
    model.addConstrs(b[k] - b_last[k] <= lambda_b * b_hat[-1] for k in range(K))
    model.addConstrs(b[k] <= U[k] for k in range(K))
    model.addConstr(c.sum('*') <= C)
    model.addConstrs(c[k] <= N[k] for k in range(K))
    model.addConstrs(c[k] <= lambda_c * C for k in range(K))

    model.Params.LogToConsole = False  # 显示求解过程
    model.optimize()

    # print("最优目标函数值：", model.objVal)
    # print('求解结果：', model.getVars())
    # for cons in model.getConstrs():
    #     print(cons.RHS)
    # print('目标函数表达式：', model.getObjective())
    b_result, c_result = np.zeros(K), np.zeros(K)
    for i in range(K):
        b_result[i] = model.getVars()[i].X
        c_result[i] = model.getVars()[i + K].X

    return b_result, c_result, model.objVal


@njit()
def coefficient_calculate(K, S, E, A, U, N, sigma_hat, beta_e, beta_a, beta_u, eta):
    constant_e, constant_a, constant_u = 0, 0, 0  # 计算三个常数项
    for k in range(K):
        tem_e, tem_a, tem_u = 0, 0, 0
        for j in range(K):
            tem_e += E[j] / N[j] * beta_e[j] * sigma_hat[k][j]
            tem_a += A[j] / N[j] * beta_a[j] * sigma_hat[k][j]
            tem_u += U[j] / N[j] * beta_u[j] * sigma_hat[k][j]
        constant_e += S[k] * tem_e
        constant_a += S[k] * tem_a
        constant_u += S[k] * tem_u

    coefficient_b, coefficient_c = np.zeros(K), np.zeros(K)  # 计算决策变量的系数
    for j in range(K):
        tem_b, tem_c = 0, 0
        for k in range(K):
            tem_b += S[k] * sigma_hat[k][j]
            tem_c += S[k] * sigma_hat[k][j]
        coefficient_b[j] = eta / N[j] * beta_u[j] * tem_b
        coefficient_c[j] = A[j] / N[j] / N[j] * (beta_u[j] - beta_a[j]) * tem_c

    return constant_e, constant_a, constant_u, coefficient_b, coefficient_c
