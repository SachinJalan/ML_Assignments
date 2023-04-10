import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import random
import pandas as pd
np.random.seed(1234)
x = np.linspace(0, 10, 500)
eps = np.random.normal(0, 5, 500)
y = x**2 +1 + eps
Xt=pd.DataFrame(x)
yt=pd.Series(y)
randomm=[i for i in range(0,len(Xt))]
random.shuffle(randomm)
X=Xt.iloc[randomm,:]
Y=yt.iloc[randomm]
Xside=Xt.iloc[0:(int(0.20*len(Xt)))]
Yside=yt.iloc[0:(int(0.20*len(Xt)))]
def kfold(X: pd.DataFrame, y: pd.Series, k, d):
    l = [(i*int(len(X)/k)) for i in range(0,  k+1)]
    arrtemp=[]
    arrtemp2=[]
    score = 0
    for i in range(len(l) - 1):
        X_valid = X.iloc[l[i] : l[i + 1]]
        y_valid = y.iloc[l[i] : l[i + 1]]
        X_train = pd.concat([X.iloc[0 : l[i]], X.iloc[l[i + 1] :]])
        y_train = pd.concat([y.iloc[0 : l[i]], y.iloc[l[i + 1] :]])
        dt = DecisionTreeRegressor(max_depth=d)
        dt.fit(X_train, y_train)
        y_hat = dt.predict(Xside)
        y_hat2=dt.predict(Xside)
        arrtemp2.append(y_hat-Yside.to_numpy())
        arrtemp.append(y_hat2)
    temp22=np.array(arrtemp)
    temp23=np.array(arrtemp2)

    varinaces=[]
    biassss=[]
    for i in range(len(temp22[0])):
        varinaces.append(np.std(temp22[:,i]))
    for i in range(len(temp23[0])):
        biassss.append(np.mean(temp23[:,i]))
    bias=np.mean(biassss)
    variance=0
    variance=np.mean(np.array(varinaces))
    return [bias,variance]
depth=[i for i in range(2,20)]
# print(depth)
bias=[]
variance=[]

for i in depth:
    a,b=kfold(X,Y,4,i)
    bias.append(a)
    variance.append(b)

# for plotting
import matplotlib.pyplot as plt
plt.plot(depth, np.array(variance), 'r-',label='variance')
plt.plot(depth, np.array(bias), 'b-',label='bias')
plt.xlabel("depth")
plt.ylabel("values")
plt.title("Bias-Variance Curve")
# plt.title(xlabel="depth")
plt.legend()
plt.show()