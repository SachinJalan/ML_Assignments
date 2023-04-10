import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree

# Or you could import sklearn DecisionTree

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 6
from sklearn.datasets import make_blobs
Xt, yt = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
X = pd.DataFrame(Xt)
y = pd.Series(yt, dtype="category")

criteria = "information_gain"
tree = DecisionTree(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
plt.show()
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))
