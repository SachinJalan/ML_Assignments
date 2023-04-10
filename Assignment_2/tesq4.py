import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree

# # Or use sklearn decision tree

# ########### BaggingClassifier ###################

from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack
from sklearn.datasets import make_blobs,make_moons
# from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
# generate dataset
from multiprocessing import Process,Queue,Pool
# define bounds of the domain
import time
X,y=make_moons(n_samples=500,noise=0.3, random_state=0)
model = BaggingClassifier(base_estimator='DecisionTree',n_estimators=20)
Xt=pd.DataFrame(X)
procs=[]
trees=[]
yt=pd.Series(y,dtype='category')
start=time.time()
model.fitparallel(Xt,yt)
end=time.time()
a=model.plot()
plt.show()
print(end-start)
start2=time.time()
