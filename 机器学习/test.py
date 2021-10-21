import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
import openpyxl as op
from openpyxl import load_workbook


def func(x): return np.sin(2*np.pi*x)


# #############################################################################
# 生成带噪声的正弦数据
size = 25
rng = np.random.RandomState(1234)  # 伪随机数生成器
x_train = rng.uniform(0., 1., size)
y_train = func(x_train) + rng.normal(scale=0.1, size=size)
x_test = np.linspace(0., 1., 100)
print(x_train,type(x_train))
print(y_train,type(y_train))
print(x_test,type(x_test))


# #############################################################################
# 拟合三次多项式
n_order = 3
X_train = np.vander(x_train, n_order + 1, increasing=True)  # 生成最高n_order次幂的，升序的，范德蒙矩阵
print(X_train,type(X_train))
X_test = np.vander(x_test, n_order + 1, increasing=True)

# #############################################################################
# 用对数边际似然 (L) 绘制真实曲线和预测曲线
reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i, ax in enumerate(axes):
    # Bayesian ridge regression with different initial value pairs
    if i == 0:
        init = [1 / np.var(y_train), 1.]  # Default values
    elif i == 1:
        init = [1., 1e-3]
        reg.set_params(alpha_init=init[0], lambda_init=init[1])
    reg.fit(X_train, y_train)
    ymean, ystd = reg.predict(X_test, return_std=True)

    ax.plot(x_test, func(x_test), color="blue", label="sin($2\\pi x$)")
    ax.scatter(x_train, y_train, s=50, alpha=0.5, label="observation")
    ax.plot(x_test, ymean, color="red", label="predict mean")
    ax.fill_between(x_test, ymean-ystd, ymean+ystd,
                    color="pink", alpha=0.5, label="predict std")
    ax.set_ylim(-1.3, 1.3)
    ax.legend()
    title = "$\\alpha$_init$={:.2f},\\ \\lambda$_init$={}$".format(
            init[0], init[1])
    if i == 0:
        title += " (Default)"
    ax.set_title(title, fontsize=12)
    text = "$\\alpha={:.1f}$\n$\\lambda={:.3f}$\n$L={:.1f}$".format(
           reg.alpha_, reg.lambda_, reg.scores_[-1])
    ax.text(0.05, -1.0, text, fontsize=12)

plt.tight_layout()  # 使子图充满区域
plt.show()
