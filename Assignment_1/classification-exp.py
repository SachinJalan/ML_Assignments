import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read dataset
# ...
# 
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
X1=pd.DataFrame(X)
y1=pd.Series(y,dtype='category')
X_train,X_test=X1[0:int(0.70*len(X1))],X1[int(0.70*len(X1)):]
y_train,y_test=y1[0:int(0.70*len(y1))],y1[int(0.70*len(y1)):]
dt=DecisionTree(max_depth=2)
dt.fit(X_train,y_train)

y_hat=dt.predict(X_test)
print(f"accuracy: {accuracy(y_hat,y_test)}")
for i in y1.unique():
    print(f"precision of {i}: {precision(y_hat,y_test,i)}")
    print(f"recall of {i}: {recall(y_hat,y_test,i)}")

def kfold(X: pd.DataFrame, y: pd.Series, k, d):
    l = [(i*int(len(X)/k)) for i in range(0,  k+1)]
    
    score = 0
    for i in range(len(l) - 1):
        X_valid = X.iloc[l[i] : l[i + 1]]
        y_valid = y.iloc[l[i] : l[i + 1]]
        X_train = pd.concat([X.iloc[0 : l[i]], X.iloc[l[i + 1] :]])
        y_train = pd.concat([y.iloc[0 : l[i]], y.iloc[l[i + 1] :]])
        dt = DecisionTree(max_depth=d)
        dt.fit(X_train, y_train)
#         dt.ans
        y_hat = dt.predict(X_valid)
        score += accuracy(y_hat, y_valid)
    return score / k


def nestoneCV(X: pd.DataFrame, y: pd.Series, k):
    accur = 0
    best = -1
    for i in range(1, 10):
        temp=kfold(X,y,k,i)
        if temp > accur:
            best = i
            accur=temp
    return best


def nestedCV(X: pd.DataFrame, y: pd.Series, outerk, innerk):
    l = [(i*int(len(X)/outerk)) for i in range(0,  outerk+1)]
    for i in range(len(l) - 1):
        # X_test = X.iloc[l[i] : l[i + 1]]
        # y_test = y.iloc[l[i] : l[i + 1]]
        X_train = pd.concat([X.iloc[0 : l[i]], X.iloc[l[i + 1] :]])
        y_train = pd.concat([y.iloc[0 : l[i]], y.iloc[l[i + 1] :]])
        optmdpth=nestoneCV(X_train, y_train, innerk)
        print(optmdpth)
        # dtt=DecisionTree()
        # dtt.fit(X_test,y_test,depth=optmdpth)
#         dtt.plot()
#         plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        # y_hat = dtt.predict(X_test)
#         print(y_hat)
#         print(y_test)
#         print(accuracy(y_hat, y_test))
nestedCV(X1,y1,5,5)