import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from sklearn.datasets import make_blobs,make_moons
from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
X,y=make_moons(n_samples=500,noise=0.3, random_state=0)
Xt=pd.DataFrame(X)
yt=pd.Series(y,dtype='category')
model=RandomForestClassifier(n_estimators=10)
model.fit(Xt,yt)
model.plot()
plt.show()
###Write code here
