"""
近似动态规划算法——ADP
之：神经网络近似
"""
import numpy as np
import math
from sklearn.neural_network import MLPRegressor


def adp_fnn(train_x, train_y, W):
    reg = MLPRegressor(hidden_layer_sizes=(W,), activation='relu', solver='lbfgs'
                       , alpha=0, tol=1e-4, max_iter=10000000)
    reg.fit(train_x, train_y)
    # print(reg.coefs_[0], reg.coefs_[0].shape)
    # print(reg.coefs_[1][:, 0], reg.coefs_[1].shape)
    # print(reg.intercepts_[0], reg.intercepts_[0].shape)
    # print(reg.intercepts_[1])

    # print(train_y)
    # print(reg.predict(train_x))

    xx = 0
    for i in range(len(train_y)):
        xx = math.fabs(train_y[i] - reg.predict(train_x)[i])
    # print(f'总绝对值误差：{xx}')
    xx /= len(train_y)
    # print(f'平均绝对值误差：{xx}')
    return reg.coefs_[0], reg.coefs_[1][:, 0], reg.intercepts_[0], reg.intercepts_[1][0], xx
