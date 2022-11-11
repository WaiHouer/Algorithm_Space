"""
近似动态规划算法——ADP
之：神经网络近似
"""
import time
import numpy as np
import math
from numba import njit
from sklearn.neural_network import MLPRegressor


def adp_fnn(train_x, train_y, W, norm_tag=None):
    reg = MLPRegressor(hidden_layer_sizes=(W,), activation='relu', solver='lbfgs'
                       , tol=1e-4, max_iter=5000)
    if norm_tag is None:
        norm_up = np.zeros(len(train_x[0, :]))  # 分子-0
        norm_down = np.ones(len(train_x[0, :]))  # 分母=1
        reg.fit(train_x, train_y)
        err = np.abs(train_y - reg.predict(train_x)) / train_y
        tag = 0
        while np.mean(err) >= 0.9:  # 重新拟合
            tag += 1
            print(f'重新拟合')
            reg.fit(train_x, train_y)
            err = np.abs(train_y - reg.predict(train_x)) / train_y
            print(np.mean(err))
            if tag == 20:
                print(f'重新拟合次数达{tag}次，停止')
                break
        print(np.mean(err))
    elif norm_tag == 'max_norm':
        train_x_, norm_up, norm_down = max_normalize(train_x)
        reg.fit(train_x_, train_y)
        err = np.abs(train_y - reg.predict(train_x_)) / train_y
        tag = 0
        while np.mean(err) >= 0.9:  # 重新拟合
            tag += 1
            print(f'重新拟合')
            reg.fit(train_x_, train_y)
            err = np.abs(train_y - reg.predict(train_x_)) / train_y
            print(np.mean(err))
            if tag == 20:
                print(f'重新拟合次数达{tag}次，停止')
                break
    elif norm_tag == 'standard_norm':
        train_x_, norm_up, norm_down = standard_normalize(train_x)
        reg.fit(train_x_, train_y)
        err = np.abs(train_y - reg.predict(train_x_)) / train_y
        tag = 0
        while np.mean(err) >= 0.9:  # 重新拟合
            tag += 1
            print(f'重新拟合')
            reg.fit(train_x_, train_y)
            err = np.abs(train_y - reg.predict(train_x_)) / train_y
            print(np.mean(err))
            if tag == 20:
                print(f'重新拟合次数达{tag}次，停止')
                break
        print(np.mean(err))

    return reg.coefs_[0], reg.coefs_[1][:, 0], reg.intercepts_[0], reg.intercepts_[1][0], norm_up, norm_down


def max_normalize(train_x):  # 极值归一化
    train_x_ = np.zeros((len(train_x), len(train_x[0, :])))
    norm_min = np.zeros(len(train_x[0, :]))
    norm_max = np.ones(len(train_x[0, :]))
    for col in range(len(train_x[0, :])):
        col_min = min(train_x[:, col])
        norm_min[col] = col_min
        col_max = max(train_x[:, col])
        norm_max[col] = col_max
        for row in range(len(train_x)):
            train_x_[row][col] = (train_x[row][col] - col_min) / (col_max - col_min)

    return train_x_, norm_min, norm_max - norm_min


def standard_normalize(train_x):
    train_x_ = np.zeros((len(train_x), len(train_x[0, :])))
    norm_mean = np.zeros(len(train_x[0, :]))
    norm_std = np.ones(len(train_x[0, :]))
    for col in range(len(train_x[0, :])):
        col_mean = np.mean(train_x[:, col])
        norm_mean[col] = col_mean
        col_std = np.std(train_x[:, col])
        norm_std[col] = col_std
        for row in range(len(train_x)):
            train_x_[row][col] = (train_x[row][col] - col_mean) / col_std

    return train_x_, norm_mean, norm_std
