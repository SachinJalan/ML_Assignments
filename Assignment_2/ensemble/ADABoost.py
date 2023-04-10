import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/sachin/github-classroom/ES654/es654-spring2023-assignment2-aaryan-sachin/tree')
from tree.base import DecisionTree2
class AdaBoostClassifier():
    trees=[]
    alphas=[]
    X_copy=pd.DataFrame()
    Y_copy=pd.Series()
    weightAll=[]
    def __init__(self, base_estimator, n_estimators=5): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.trees=[]
        self.alphas=[]
        self.X_copy=pd.DataFrame()
        self.Y_copy=pd.Series()
        self.weightAll=[]

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X_copy=X
        self.Y_copy=y
        val=1/len(X)
        weights=[val for i in range(len(X))]
        weights=pd.Series(weights)
        for i in range(self.n_estimators):
            model=DecisionTree2(max_depth=1)
            model.fit(X,y,weights)
            self.trees.append(model)
            yh1=model.predict(X)
            self.weightAll.append((weights*(10/np.mean(weights))).to_numpy())
            err=(np.sum(weights[yh1!=y])/np.sum(weights))
            alpha=0.5*np.log((1-err)/err)
            self.alphas.append(alpha)
            weights[yh1==y]=weights[yh1==y].apply(lambda x: x*np.exp(-alpha))
            weights[yh1!=y]=weights[yh1!=y].apply(lambda x:x*np.exp(alpha))
            totw=np.sum(weights)
            weights=weights/totw

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predicted=[]
        for i in range(len(self.trees)):
            yhat=self.trees[i].predict(X)
            yhat[yhat==0]=-1
            
            predicted.append(self.alphas[i]*(yhat.to_numpy()))
        df2=pd.DataFrame(predicted)
        df3=(df2.sum(axis=0))/(len(X))
        df3[df3>0]=int(1)
        df3[df3<=0]=int(0)
        # df3.dtype=int
        df3=df3.apply(lambda x: (int(x)))
        return df3
        

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        X=self.X_copy.to_numpy()
        y=self.Y_copy.to_numpy()
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
        # print("hello")
        fig1,ax=plt.subplots(1, self.n_estimators, sharex='col', sharey='row',figsize=(6*(self.n_estimators),6))
        # define the model
        for i in range(self.n_estimators):
            model = self.trees[i]
            # fit the model
            Xt=pd.DataFrame(X)
            yt=pd.Series(y,dtype='category')
            temp=1/(len(Xt))
            # make predictions for the grid
            yhat = model.predict(pd.DataFrame(grid))
            # print(yhat)
            yhat=yhat.to_numpy()
            t=yt.unique()
            c=1;
            for ic in t:
                if(ic not in yhat):
                    yhat[-c]=ic
                    c+=1;
            # reshape the predictions back into a grid
            zz = yhat.reshape(xx.shape)
            ax[i].set_title(label=f'alpha = {self.alphas[i]}',y=-0.1)
            
            ax[i].contourf(xx,yy,zz,cmap='Paired')
            for cv in list(yt.cat.categories):
                row_ix=np.where(y==cv)
                # print(row_ix)
                ax[i].scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired',s=self.weightAll[i][row_ix])
            
        fig2,ax2=plt.subplots()
        yhat = self.predict(pd.DataFrame(grid))
        yhat=yhat.to_numpy()
        # reshape the predictions back into a grid
        zz = yhat.reshape(xx.shape)
        ax2.set_title(label='Combined Model',y=-0.1)
        ax2.contourf(xx,yy,zz,cmap='Paired')
        for cv in list(yt.cat.categories):
            row_ix=np.where(y==cv)
            # print(row_ix)
            ax2.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')

        return [fig1,fig2]
