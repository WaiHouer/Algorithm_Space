"""
近似动态规划算法——ADP
之：当期整数规划模型求解
"""
import numpy as np
import gurobipy as gp
from numba import njit


def adp_model(K, S, E, A, U, b_last, N, sigma_hat, beta_e, beta_a, beta_u, eta, b_hat, lambda_b, C, lambda_c
              , W, theta_1, phi_1, theta_2, phi_2
              , alpha, delta_a, delta_u, gamma_a, gamma_u, p, q
              , norm_up, norm_down, re_tag=None, B_add=0):
    model = gp.Model()  # 建立模型

    b = model.addVars(K, lb=0, vtype=gp.GRB.INTEGER)  # 规定变量
    c = model.addVars(K, lb=0, vtype=gp.GRB.INTEGER)
    act = model.addVars(W, lb=0, vtype=gp.GRB.CONTINUOUS)  # 规定辅助变量-活性值
    h_nxt = model.addVars(5 * K, lb=0, vtype=gp.GRB.CONTINUOUS)  # 规定中间变量，下一期状态
    miu = model.addVars(W, vtype=gp.GRB.BINARY)  # 规定0-1辅助变量
    model.update()  # 更新变量空间

    constant_e, constant_a, constant_u, coefficient_b, coefficient_c = coefficient_calculate(K, S, E, A, U, N, sigma_hat
                                                                                             , beta_e, beta_a, beta_u
                                                                                             , eta)
    model.setObjective(constant_e + constant_a + constant_u
                       - gp.quicksum(b[j] * coefficient_b[j] for j in range(K))
                       + gp.quicksum(c[j] * coefficient_c[j] for j in range(K))
                       + gp.quicksum(act[w] * theta_2[w] for w in range(W))
                       + phi_2
                       , gp.GRB.MINIMIZE)  # 规定目标函数

    B = sum(b_hat) + B_add  # 约束1（其实不算是约束）
    model.addConstr(b.sum('*') <= B)
    model.addConstrs(b[k] >= b_last[k] for k in range(K))
    model.addConstrs(b[k] - b_last[k] <= lambda_b * b_hat[-1] for k in range(K))
    model.addConstrs(eta * b[k] <= U[k] + A[k] / N[k] * c[k] for k in range(K))
    model.addConstr(c.sum('*') <= C)
    model.addConstrs(c[k] <= N[k] for k in range(K))
    model.addConstrs(c[k] <= lambda_c * C for k in range(K))

    for o in range(0, K):
        model.addConstr(h_nxt[o] == S[o] - (constant_e + constant_a + constant_u
                                            - gp.quicksum(b[j] * coefficient_b[j] for j in range(K))
                                            + gp.quicksum(c[j] * coefficient_c[j] for j in range(K))))
    for o in range(K, 2 * K):
        model.addConstr(h_nxt[o] == E[o - K] + (constant_e + constant_a + constant_u
                                                - gp.quicksum(b[j] * coefficient_b[j] for j in range(K))
                                                + gp.quicksum(c[j] * coefficient_c[j] for j in range(K)))
                        - alpha[o - K] * E[o - K])
    for o in range(2 * K, 3 * K):
        model.addConstr(h_nxt[o] == (1 - delta_a[o - 2 * K] - gamma_a[o - 2 * K]) * A[o - 2 * K]
                        + p * alpha[o - 2 * K] * E[o - 2 * K]
                        - (1 - delta_a[o - 2 * K] - gamma_a[o - 2 * K]) * A[o - 2 * K] / N[o - 2 * K] * c[o - 2 * K])
    for o in range(3 * K, 4 * K):
        model.addConstr(h_nxt[o] == (1 - delta_u[o - 3 * K] - gamma_u[o - 3 * K]) * U[o - 3 * K]
                        + (1 - p) * (1 - q) * alpha[o - 3 * K] * E[o - 3 * K]
                        + (1 - delta_u[o - 3 * K] - gamma_u[o - 3 * K]) * A[o - 3 * K] / N[o - 3 * K] * c[o - 3 * K]
                        - (1 - delta_u[o - 3 * K] - gamma_u[o - 3 * K]) * eta * b[o - 3 * K])
    for o in range(4 * K, 5 * K):
        model.addConstr(h_nxt[o] == b[o - 4 * K])

    model.addConstrs(act[w] >= gp.quicksum((h_nxt[o] - norm_up[o]) / norm_down[o] * theta_1[o][w]
                                           for o in range(5 * K)) + phi_1[w] for w in range(W))
    model.addConstrs(act[w] >= 0 for w in range(W))
    model.addConstrs(act[w] <= gp.quicksum((h_nxt[o] - norm_up[o]) / norm_down[o] * theta_1[o][w]
                                           for o in range(5 * K)) + phi_1[w]
                     + miu[w] * 9999999999999 for w in range(W))
    model.addConstrs(act[w] <= 0 + (1 - miu[w]) * 9999999999999 for w in range(W))

    # model.addConstr(gp.quicksum(b[k] for k in range(K)) == B)  # 我自己后加的，用于确保病床没有浪费

    model.Params.LogToConsole = False  # 显示求解过程
    model.optimize()

    # print("最优目标函数值：", model.objVal)
    # print('求解结果：', model.getVars())
    if model.status != gp.GRB.OPTIMAL:  # 检查出错的约束
        print('出错了')
        model.computeIIS()
        model.write('model.ilp')

    if model.getVars()[0] is None:
        for cons in model.getConstrs():
            print(cons.RHS)
    # print('目标函数表达式：', model.getObjective())
    b_result, c_result = np.zeros(K), np.zeros(K)
    for i in range(K):
        b_result[i] = model.getVars()[i].X
        c_result[i] = model.getVars()[i + K].X

    value_ADP = value_calculate(K, constant_e, constant_a, constant_u, coefficient_b, coefficient_c
                                , b_result, c_result)

    if re_tag:
        return b_result, c_result, model.objVal
    else:
        return b_result, c_result, value_ADP


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


def value_calculate(K, constant_e, constant_a, constant_u, coefficient_b, coefficient_c, b, c):
    value_ADP = constant_e + constant_a + constant_u
    for j in range(K):
        value_ADP -= coefficient_b[j] * b[j]
        value_ADP += coefficient_c[j] * c[j]
    return value_ADP
