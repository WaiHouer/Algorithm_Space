import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

X = np.linspace(-3.14, 3.14, 400)
X1 = X.reshape(-1,1)
y = np.sin(X) + 0.3*np.random.rand(len(X))

clf = MLPRegressor(alpha=1e-6,hidden_layer_sizes=(3, 2), random_state=1, max_iter=100000,activation='logistic')
clf.fit(X1, y)
MLPRegressor(activation='logistic', alpha=1e-06, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(3, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=100000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
y2 = clf.predict(X1)
plt.scatter(X,y)#画图
plt.plot(X,y2,c="red")
