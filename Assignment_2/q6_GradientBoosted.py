import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.gradientBoosted import GradientBoostedRegressor
from tree.base import DecisionTree

# Or use sklearn decision tree

########### GradientBoostedClassifier ###################
from sklearn.datasets import make_regression

X, y= make_regression(
    n_features=3,
    n_informative=3,
    noise=10,
    tail_strength=10,
    random_state=42,
)
model=GradientBoostedRegressor('DecisionTree',n_estimators=20,learning_rate=0.4)
Xt=pd.DataFrame(X)
yt=pd.Series(y)
model.fit(Xt,yt)
yhat=model.predict(Xt)
print(rmse(yhat,yt))
# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], y)
plt.show()