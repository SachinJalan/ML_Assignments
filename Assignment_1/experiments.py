
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions
X=np.arange(10,100,20)
#REAL INPUT REAL OUTPUT
def retTrainTime(N,P):
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    st=time.time()
    dt=DecisionTree(max_depth=7)
    dt.fit(X,y)
    et=time.time()
    return (et-st)
X=list(X)
mean=[]
std=[]
for i in X:
    y2=[]
    for j in range(20):
        y2.append(retTrainTime(i,5))
    sery2=pd.Series(y2)
    mean.append(sery2.mean())
    std.append(sery2.std())
plt.plot(X,mean)
plt.show()
# REAL INPUT DISCRETE OUTPUT
def retTrainTime(N,P):
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randint(P, size = N), dtype="category")
    st=time.time()
    dt=DecisionTree(max_depth=7)
    dt.fit(X,y)
    et=time.time()
    return (et-st)
X=list(X)
mean=[]
std=[]
for i in X:
    y2=[]
    for j in range(20):
        y2.append(retTrainTime(i,5))
    sery2=pd.Series(y2)
    mean.append(sery2.mean())
    std.append(sery2.std())
plt.plot(X,mean)
plt.show()
# DISCRETE INPUT DISCRETE OUTPUT
def retTrainTime(N,P):
    X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(P)})
# y = pd.Series(np.random.randint(P, size = N),  dtype="category")
    y = pd.Series(np.random.randint(P, size = N), dtype="category")
    st=time.time()
    dt=DecisionTree(max_depth=7)
    dt.fit(X,y)
    et=time.time()
    return (et-st)
X=list(X)
mean=[]
std=[]
for i in X:
    y2=[]
    for j in range(20):
        y2.append(retTrainTime(i,5))
    sery2=pd.Series(y2)
    mean.append(sery2.mean())
    std.append(sery2.std())
plt.plot(X,mean)
plt.show()
# DISCRETE INPUT REAL OUTPUT
def retTrainTime(N,P):
    X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(P)})
# y = pd.Series(np.random.randint(P, size = N),  dtype="category")
    y = pd.Series(np.random.randint(P, size = N), dtype="category")
    st=time.time()
    dt=DecisionTree(max_depth=7)
    dt.fit(X,y)
    et=time.time()
    return (et-st)
X=list(X)
mean=[]
std=[]
for i in X:
    y2=[]
    for j in range(20):
        y2.append(retTrainTime(i,5))
    sery2=pd.Series(y2)
    mean.append(sery2.mean())
    std.append(sery2.std())
plt.plot(X,mean)
plt.show()