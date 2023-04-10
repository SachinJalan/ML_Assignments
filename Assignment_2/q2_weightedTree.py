# from tree.base import DecisionTree2

# from sklearn.tree import DecisionTreeClassifier

# ## compare both the trees
# from sklearn.datasets import make_classification
# X, y = make_classification(
# n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
from tree.base import DecisionTree2
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_blobs
import random
import numpy as np
import pandas as pd
from metrics import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
# generate dataset
X, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=1, cluster_std=3)
# create scatter plot for samples from each class
# for class_value in range(2):
# 	# get row indexes for samples with this class
# 	row_ix = where(y == class_value)
# 	# create scatter of these samples
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # show the plot
# pyplot.show()
Xt=pd.DataFrame(X)
yt=pd.Series(y,dtype='category')
weights=[random.random() for i in range(len(Xt))]
weight=pd.Series(weights)
s=weight.sum()
weight=weight/s;
Xtrain=Xt.iloc[0:int(0.70*len(Xt))]
Ytrain=yt.iloc[0:int(0.70*len(Xt))]
Xtest=Xt.iloc[int(0.70*len(Xt)):]
Ytest=yt.iloc[int(0.70*len(Xt)):]
weightT=weight[0:int(0.70*len(Xt))]
model=DecisionTree2(max_depth=5)
model2=DecisionTreeClassifier(max_depth=5)
min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
# define the x and y scale
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)
# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))
model.fit(Xtrain,Ytrain,weightT)
yhat=model.predict(Xtest)
model2.fit(Xtrain,Ytrain,sample_weight=weightT)
yhat3=model2.predict(Xtest)
print("accuracy o my tree = ",accuracy(yhat,Ytest))
print("accuracy of sklearn = ",accuracy(pd.Series(yhat3),Ytest))
fig2,ax2=plt.subplots()
yhat2 = model.predict(pd.DataFrame(grid))
yhat2=yhat2.to_numpy()
# reshape the predictions back into a grid
zz = yhat2.reshape(xx.shape)
ax2.set_title(label='Decision Surface',y=-0.15)
ax2.contourf(xx,yy,zz,cmap='Paired')
for cv in list(yt.cat.categories):
    row_ix=np.where(y==cv)
    # print(row_ix)
    ax2.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired',s=8000*weight.to_numpy()[row_ix])
plt.show()