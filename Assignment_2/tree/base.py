"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain_Real, gini_index,entropy2
from dataclasses import dataclass
import random
from tree.utils import *
np.random.seed(42)


@dataclass
class DecisionTree:
#     criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
#     max_depth : int
#     maxdepth=10# The maximum depth the tree can grow to
    X_copy=pd.DataFrame()
    Y_copy=pd.Series()
    ans = dict()
    def __init__(self,max_depth=100,criterion="information_gain"):
        self.max_depth=max_depth
        self.criterion=criterion
        self.X_copy=pd.DataFrame()
        self.Y_copy=pd.Series()

    def DisInDisOut(X: pd.DataFrame, y: pd.Series, depth,flag) -> None:
        d = {}
        if entropy(y) == 0:
            return y.iloc[0]
        if depth == 0:
            return y.value_counts().index[0]
        m = 0
        ind = 0
        for i in range(0, len(X.columns)):
            if information_gain(y, X.iloc[:, i],flag) > m:
                ind = i
                m = information_gain(y, X.iloc[:, i],flag)
        key = X.columns[ind]
        pard = {}
        for i in X.iloc[:, ind].unique():
            example = X.loc[X[X.columns[ind]] == i, :]
            examples = example.drop([X.columns[ind]], axis=1)
            ynew = y[X[X.columns[ind]] == i]
            if examples.empty:
                pard[i] = ynew.value_counts().index[0]
            else:
                pard[i] = DecisionTree.DisInDisOut(
                    pd.DataFrame(examples), ynew, depth - 1,flag
                )
        d[key] = pard
        return d

    def DisInRealOut(X: pd.DataFrame, y: pd.Series, depth) -> None:
        d = {}
        m = float("-inf")
        ind = 0
        if depth == 0:
            return y.mean()
        for i in range(0, len(X.columns)):
            if information_gain_DIRO(y, X.iloc[:, i]) > m:
                ind = i
                m = information_gain_DIRO(y, X.iloc[:, i])
        key = X.columns[ind]
        pard = {}
        # print(m)
#         if m < 0:
#             return y.mean()

        for i in X.iloc[:, ind].cat.categories:
            example = X.loc[X[X.columns[ind]] == i, :]
            if(example.empty):
                pard[i]=y.mean()
                continue
            examples = example.drop([X.columns[ind]], axis=1)
            ynew = y[X[X.columns[ind]] == i]
            if examples.empty:
                pard[i] = ynew.mean()
            else:
                pard[i] = DecisionTree.DisInRealOut(
                    pd.DataFrame(examples), ynew, depth - 1
                )
        d[key] = pard
        return d

    def fit_RIDO(X: pd.DataFrame, y: pd.Series, depth,flag) -> None:
        """
        Function to train and construct the decision tree
        """
        d = {}
        if entropy(y) == 0:
            return y.iloc[0]
        if depth == 0:
            return y.value_counts().index[0]
        m = -2
        ind = 0
        val = 0
        for i in range(0, len(X.columns)):
            IGR = information_gain_Real(y, X.iloc[:, i],flag)
            if IGR[0] > m:
                ind = i
                m = IGR[0]
                val = IGR[1]
        key = X.columns[ind]
        if m < 0:
            return y.value_counts().index[0]
        pard = {
            f"Less Than {val}": DecisionTree.fit_RIDO(
                pd.DataFrame(X[X.iloc[:, ind] < val]),
                y[X.iloc[:, ind] < val],
                depth - 1,flag
            ),
            f"Greater Than {val}": DecisionTree.fit_RIDO(
                pd.DataFrame(X[X.iloc[:, ind] >= val]),
                y[X.iloc[:, ind] >= val],
                depth - 1,flag
            ),
        }
        d[key] = pard
        return d

    def fit_RIRO(X: pd.DataFrame, y: pd.Series, depth) -> None:
        """
        Function to train and construct the decision tree
        """
        d = {}
        if mse(y) == 0:
            return y.mean()
        if depth == 0:
            return y.mean()

        m = -2
        ind = 0
        val = 0
        for i in range(0, len(X.columns)):
            IGR = information_gain_Real2(y, X.iloc[:, i])
            if IGR[0] > m:
                ind = i
                m = IGR[0]
                val = IGR[1]
        key = X.columns[ind]
        if m < 0:
            return y.mean()
        pard = {
            f"Less Than {val}": DecisionTree.fit_RIRO(
                pd.DataFrame(X[X.iloc[:, ind] < val]),
                y[X.iloc[:, ind] < val],
                depth - 1,
            ),
            f"Greater Than {val}": DecisionTree.fit_RIRO(
                pd.DataFrame(X[X.iloc[:, ind] >= val]),
                y[X.iloc[:, ind] >= val],
                depth - 1,
            ),
        }
        d[key] = pard
        return d

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.X_copy=X
        self.Y_copy=y
        if X.iloc[:,0].dtype == "category":
            if y.dtype == "category":
                if(self.criterion=="information_gain"):
                    self.ans = DecisionTree.DisInDisOut(X, y,self.max_depth,1)
                else:
                    self.ans = DecisionTree.DisInDisOut(X, y,self.max_depth,0)
            else:
                self.ans = DecisionTree.DisInRealOut(X, y, self.max_depth)
        else:
            if y.dtype == "category":
                if(self.criterion=="information_gain"):
                    self.ans = DecisionTree.fit_RIDO(X, y, self.max_depth,1)
                else:
                    self.ans = DecisionTree.fit_RIDO(X, y, self.max_depth,0)
            else:
                self.ans = DecisionTree.fit_RIRO(X, y, self.max_depth)

    def predReal(ans, X: pd.Series):
        if type(ans) != dict:
            return ans
        for i in ans.keys():
            if type(ans[i]) == dict:
                for j in ans[i].keys():
                    if j.split(" ")[0] == "Less" and X[i] <= float(j.split(" ")[2]):
                        if type(ans[i][j]) == dict:
                            return DecisionTree.predReal(ans[i][j], X)
                        else:
                            return ans[i][j]
                    elif j.split(" ")[0] == "Greater" and X[i] >= float(
                        j.split(" ")[2]
                    ):
                        if type(ans[i][j]) == dict:
                            return DecisionTree.predReal(ans[i][j], X)
                        else:
                            return ans[i][j]
            else:
                return ans[i][X[i]]

    def predDiscrete(ans, X: pd.Series):
        if type(ans) != dict:
            return ans
        for i in ans.keys():
#             print(f"working {ans[i]}")
            if type(ans[i][X[i]]) == dict:
                return DecisionTree.predDiscrete(ans[i][X[i]], X)
            else:
                return ans[i][X[i]]
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        if(X.iloc[:,0].dtype=='category'):
            y_hat=[];
            for i in range(len(X)):
#                 print(f"processing{i}")
                y_hat.append(DecisionTree.predDiscrete(self.ans,X.iloc[i]))
#                 print(y_hat[i])
        else:
            y_hat=[]
            for i in range(len(X)):
                y_hat.append(DecisionTree.predReal(self.ans,X.iloc[i]))
        return pd.Series(y_hat,index=X.index)
    def plot_h(space,d) -> None:
        if(type(d)!=dict):
            return d
        else:
            for i in d.keys():
                if(type(d[i])==dict):
                    for j in d[i].keys():
                        print(" "*space +f"If {i} is {j}")
                        if (type(d[i][j])==dict):
                            DecisionTree.plot_h(space+4,d[i][j])
                        else:
                            print(" "*(space+4) + f"ans={d[i][j]}")
    def plot(self):
        DecisionTree.plot_h(0,self.ans)

class DecisionTree2:
#     criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
#     max_depth : int
#     maxdepth=10# The maximum depth the tree can grow to
    ans = dict()
    def __init__(self,max_depth=100,criterion="information_gain"):
        self.max_depth=max_depth
        self.criterion=criterion
    
    def fit_RIDO(X: pd.DataFrame, y: pd.Series, weight: pd.Series,depth,flag) -> None:
        """
        Function to train and construct the decision tree
        """
        d = {}
        if entropy2(y,weight) == 0:
            return y.iloc[0]
        if depth == 0:
            return y.value_counts().index[0]
        m = -2
        ind = 0
        val = 0
        for i in range(0, len(X.columns)):
            IGR = information_gain_RealAda(y, X.iloc[:, i],weight,flag)
            if IGR[0] > m:
                ind = i
                m = IGR[0]
                val = IGR[1]
        key = X.columns[ind]
        if m < 0:
            return y.value_counts().index[0]
        pard = {
            f"Less Than {val}": DecisionTree2.fit_RIDO(
                pd.DataFrame(X[X.iloc[:, ind] < val]),
                y[X.iloc[:, ind] < val],weight[X.iloc[:, ind] < val],
                depth - 1,flag
            ),
            f"Greater Than {val}": DecisionTree2.fit_RIDO(
                pd.DataFrame(X[X.iloc[:, ind] >= val]),
                y[X.iloc[:, ind] >= val],weight[X.iloc[:, ind] >= val],
                depth - 1,flag
            ),
        }
        d[key] = pard
        return d
    def fit(self, X: pd.DataFrame, y: pd.Series,weight: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        if y.dtype == "category":
            if(self.criterion=="information_gain"):
                self.ans = DecisionTree2.fit_RIDO(X, y,weight, self.max_depth,1)
            else:
                self.ans = DecisionTree2.fit_RIDO(X, y,weight, self.max_depth,0)

    def predReal(ans, X: pd.Series):
        if type(ans) != dict:
            return ans
        for i in ans.keys():
            if type(ans[i]) == dict:
                for j in ans[i].keys():
                    if j.split(" ")[0] == "Less" and X[i] <= float(j.split(" ")[2]):
                        if type(ans[i][j]) == dict:
                            return DecisionTree2.predReal(ans[i][j], X)
                        else:
                            return ans[i][j]
                    elif j.split(" ")[0] == "Greater" and X[i] >= float(
                        j.split(" ")[2]
                    ):
                        if type(ans[i][j]) == dict:
                            return DecisionTree2.predReal(ans[i][j], X)
                        else:
                            return ans[i][j]
            else:
                return ans[i][X[i]]
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
#         if(X.iloc[:,0].dtype=='category'):
#             y_hat=[];
#             for i in range(len(X)):
# #                 print(f"processing{i}")
#                 y_hat.append(DecisionTree2.predDiscrete(self.ans,X.iloc[i]))
# #                 print(y_hat[i])
#         else:
        y_hat=[]
        for i in range(len(X)):
            y_hat.append(DecisionTree2.predReal(self.ans,X.iloc[i]))
        return pd.Series(y_hat,index=X.index)
@dataclass
class DecisionTreeRF:
#     criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
#     max_depth : int
#     maxdepth=10# The maximum depth the tree can grow to
    data=[]
    ans = dict()
    def __init__(self,max_depth=100,criterion="information_gain"):
        self.max_depth=max_depth
        self.criterion=criterion
        self.data=[]
    def fit_RIDORF(X: pd.DataFrame, y: pd.Series, depth,flag) -> None:
        """
        Function to train and construct the decision tree
        """
        d = {}
        if entropy(y) == 0:
            return y.iloc[0]
        if depth == 0:
            return y.value_counts().index[0]
        m = -2
        ind = 0
        val = 0
        l=len(list(X.columns))
        lis=[i for i in range(0,l)]
        features=random.sample(lis,int(l/2))
        
        for i in features:
            IGR = information_gain_Real(y, X.iloc[:, i],flag)
            if IGR[0] > m:
                ind = i
                m = IGR[0]
                val = IGR[1]
        key = X.columns[ind]
        if m < 0:
            return y.value_counts().index[0]
        pard = {
            f"Less Than {val}": DecisionTreeRF.fit_RIDORF(
                pd.DataFrame(X[X.iloc[:, ind] < val]),
                y[X.iloc[:, ind] < val],
                depth - 1,flag
            ),
            f"Greater Than {val}": DecisionTreeRF.fit_RIDORF(
                pd.DataFrame(X[X.iloc[:, ind] >= val]),
                y[X.iloc[:, ind] >= val],
                depth - 1,flag
            ),
        }
        d[key] = pard
        return d

    def fit_RIRORF(X: pd.DataFrame, y: pd.Series, depth) -> None:
        """
        Function to train and construct the decision tree
        """
        d = {}
        if mse(y) == 0:
            return y.mean()
        if depth == 0:
            return y.mean()

        m = -2
        ind = 0
        val = 0
        l=len(list(X.columns))
        lis=[i for i in range(0,l)]
        features=random.sample(lis,int(l/2))
        for i in features:
            IGR = information_gain_Real2(y, X.iloc[:, i])
            if IGR[0] > m:
                ind = i
                m = IGR[0]
                val = IGR[1]
        key = X.columns[ind]
        if m < 0:
            return y.mean()
        pard = {
            f"Less Than {val}": DecisionTreeRF.fit_RIRORF(
                pd.DataFrame(X[X.iloc[:, ind] < val]),
                y[X.iloc[:, ind] < val],
                depth - 1,
            ),
            f"Greater Than {val}": DecisionTreeRF.fit_RIRORF(
                pd.DataFrame(X[X.iloc[:, ind] >= val]),
                y[X.iloc[:, ind] >= val],
                depth - 1,
            ),
        }
        d[key] = pard
        return d

    def fitRF(self, X: pd.DataFrame, y: pd.Series) -> None:
        lbag=set()
        # indices=[i for i in range(len(X))]
        for i in range(int(len(X)/3)):
            lbag.add(random.randint(0,len(X)-1))
        lbag2=list(lbag)
        X=X.iloc[lbag2]
        y=y.iloc[lbag2]
        self.data=lbag2
        if y.dtype == "category":
            if(self.criterion=="information_gain"):
                self.ans = DecisionTreeRF.fit_RIDORF(X, y, self.max_depth,1)
            else:
                self.ans = DecisionTreeRF.fit_RIDORF(X, y, self.max_depth,0)
        else:
            self.ans = DecisionTreeRF.fit_RIRORF(X, y, self.max_depth)

    def predReal(ans, X: pd.Series):
        if type(ans) != dict:
            return ans
        for i in ans.keys():
            if type(ans[i]) == dict:
                for j in ans[i].keys():
                    if j.split(" ")[0] == "Less" and X[i] <= float(j.split(" ")[2]):
                        if type(ans[i][j]) == dict:
                            return DecisionTreeRF.predReal(ans[i][j], X)
                        else:
                            return ans[i][j]
                    elif j.split(" ")[0] == "Greater" and X[i] >= float(
                        j.split(" ")[2]
                    ):
                        if type(ans[i][j]) == dict:
                            return DecisionTreeRF.predReal(ans[i][j], X)
                        else:
                            return ans[i][j]
            else:
                return ans[i][X[i]]

    def predDiscrete(ans, X: pd.Series):
        if type(ans) != dict:
            return ans
        for i in ans.keys():
#             print(f"working {ans[i]}")
            if type(ans[i][X[i]]) == dict:
                return DecisionTreeRF.predDiscrete(ans[i][X[i]], X)
            else:
                return ans[i][X[i]]
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        if(X.iloc[:,0].dtype=='category'):
            y_hat=[];
            for i in range(len(X)):
#                 print(f"processing{i}")
                y_hat.append(DecisionTreeRF.predDiscrete(self.ans,X.iloc[i]))
#                 print(y_hat[i])
        else:
            y_hat=[]
            for i in range(len(X)):
                y_hat.append(DecisionTreeRF.predReal(self.ans,X.iloc[i]))
        return pd.Series(y_hat,index=X.index)
    def plot_h(space,d) -> None:
        if(type(d)!=dict):
            return d
        else:
            for i in d.keys():
                if(type(d[i])==dict):
                    for j in d[i].keys():
                        print(" "*space +f"If {i} is {j}")
                        if (type(d[i][j])==dict):
                            DecisionTreeRF.plot_h(space+4,d[i][j])
                        else:
                            print(" "*(space+4) + f"ans={d[i][j]}")
    def plot(self):
        DecisionTreeRF.plot_h(0,self.ans)