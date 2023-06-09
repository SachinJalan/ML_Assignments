import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
# from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)

########### RandomForestClassifier ###################

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['information_gain']:
    Classifier_RF = RandomForestClassifier(n_estimators=10, criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot()
    # plt.show()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

Regressor_RF = RandomForestRegressor(n_estimators=10)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
plt.show()
print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
