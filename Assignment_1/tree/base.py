from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tree.utils import entropy, information_gain, gini_index,information_gain_DIRO,information_gain_Real,information_gain_Real2,mse
np.random.seed(42)


@dataclass
class DecisionTree:
#     criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
#     max_depth : int
#     maxdepth=10# The maximum depth the tree can grow to
    ans = dict()
    def __init__(self,max_depth=100,criterion="information_gain"):
        self.max_depth=max_depth
        self.criterion=criterion

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