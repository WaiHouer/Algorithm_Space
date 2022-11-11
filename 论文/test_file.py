import numpy as np
import math
from scipy.optimize import minimize


class SLSQP_Method:
    def __init__(self):
        self.method = 'SLSQP'
        self.x = np.zeros(3)
        self.algorithm()

    def function(self):
        func = lambda x: 10 * x[0] * x[1] + x[2]
        return func

    def constraints(self):
        cons = ()
        for i in range(3):
            cons += ({'type': 'ineq', 'fun': lambda x: x[i] - i},)
        return cons

    def algorithm(self):
        print(self.function(), self.constraints())
        res = minimize(fun=self.function(), x0=self.x, method=self.method, constraints=self.constraints())
        print(res.fun)
        print(res.success)
        print(res.x)


if __name__ == '__main__':
    SLSQP_Method()
