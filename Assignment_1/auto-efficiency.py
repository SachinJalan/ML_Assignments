
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read real-estate data set
# ...
# 
df=pd.read_csv('modified_automobile_data.csv')
data=df.iloc[:,1:]
for i in range(1,8):
    data.iloc[:,i]=data.iloc[:,i].astype('category')
Xtrain,Xtest=data.iloc[0:int(0.70*len(data)),1:],data.iloc[int(0.70*len(data)):,1:]
ytrain,ytest=data.iloc[0:int(0.70*len(data)),0],data.iloc[int(0.70*len(data)):,0]
dt=DecisionTree(max_depth=4)
dt.fit(Xtrain,ytrain)
print(f"rmse by my model: {rmse(dt.predict(Xtest),ytest)}")
from sklearn.tree import DecisionTreeRegressor
skdt=DecisionTreeRegressor(max_depth=4)
skdt.fit(Xtrain,ytrain)
y_hat2=pd.Series(skdt.predict(Xtest))
print(f"rmse by sklearn model: {rmse(y_hat2,ytest)}")

