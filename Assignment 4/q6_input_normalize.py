from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import numpy as np
from linearRegression.linear_regression import LinearRegression
import matplotlib.pyplot as plt
num=np.arange(10,10000,1000)
err1=[]
err2=[]
for i in num:
    np.random.seed(45)

    N = i
    P = 3 
    x = np.random.randint(1000*N*P, size=(N, P))
    y = np.array(np.random.randint(N, size=(N,1)))
    # X[1]=X[1]*1000
    
    model=LinearRegression()
    model.fit_normal_equations(x,y)
    err11=model.mse_loss(x,y,model.coef)
    # print(y-x@model.coef)
    scaler=StandardScaler()
    scaler.fit(x)
    X2=scaler.transform(x)
    model2=LinearRegression()
    model2.fit_normal_equations(X2,y)
    err22=model2.mse_loss(X2,y,model2.coef)
    err1.append(err11)
    err2.append(err22)
fig,ax=plt.subplots()
plt.plot(num,err1,'k')
plt.plot(num,err2,'b')
plt.show()