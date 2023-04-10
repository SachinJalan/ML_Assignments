from tree.base import DecisionTree
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class GradientBoostedRegressor:
    trees=[]
    average=0
    def __init__(
        self, base_estimator, n_estimators=3, learning_rate=0.1
    ):
          # Optional Arguments: Type of estimator
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        :param learning_rate: The learning rate shrinks the contribution of each tree by `learning_rate`.
        """
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.trees=[]
        self.average=0

    def fit(self, X, y):
        """
        Function to train and construct the GradientBoostedRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        self.average=y.mean()
        average=y.mean()
        residue=y-average
        for i in range(self.n_estimators):
            tree=DecisionTree(max_depth=5)
            tree.fit(X,residue)
            jk=self.learning_rate*(tree.predict(X))
            for j in self.trees:
                jk+=self.learning_rate*(j.predict(X))
            self.trees.append(tree)
            jk+=average
            residue=y-jk

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        ltemp=[self.average for i in range(len(X))]
        yhat=pd.Series(ltemp)
        for i in self.trees:
            yhat+=(self.learning_rate*i.predict(X))
        return yhat
