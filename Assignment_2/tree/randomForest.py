from .base import *
from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack
import numpy as np
import matplotlib.pyplot as plt
class RandomForestClassifier():
    trees=[]
    X_copy=pd.DataFrame()
    Y_copy=pd.Series()
    def __init__(self, n_estimators=100, criterion='gini', max_depth=100):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.max_depth=max_depth
        self.trees=[]
        self.X_copy=pd.DataFrame()
        self.Y_copy=pd.Series()

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X_copy=X
        self.Y_copy=y
        for i in range(self.n_estimators):
            model=DecisionTreeRF(max_depth=self.max_depth)
            model.fitRF(X,y)
            self.trees.append(model)
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        l=[]
        for i in self.trees:
            l2=i.predict(X).tolist()
            l.append(l2)
        ans=[]
        for i in range(len(l[0])):
            d={}
            for j in range(len(l)):
                if(l[j][i] in d):
                    d[l[j][i]]+=1
                else:
                    d[l[j][i]]=0
            ans.append(max(zip(d.values(), d.keys()))[1])
        return pd.Series(ans,dtype='category')

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        # self.fit(Xt, yt)
# model.trees
# model.plot()
        fig1,ax=plt.subplots(1, self.n_estimators, sharex='col', sharey='row',figsize=(6*(self.n_estimators),6))
        count=0
        Xt=self.X_copy
        yt=self.Y_copy
        X=Xt.to_numpy()
        y=yt.to_numpy()
        for i in self.trees:
            min1, max1 = Xt.iloc[i.data, 0].min()-1, Xt.iloc[i.data, 0].max()+1
            min2, max2 = Xt.iloc[i.data, 1].min()-1, Xt.iloc[i.data, 1].max()+1
            # define the x and y scale
            x1grid = arange(min1, max1, 0.1)
            x2grid = arange(min2, max2, 0.1)
            # create all of the lines and rows of the grid
            xx, yy = meshgrid(x1grid, x2grid)
            # flatten each grid to a vector
            r1, r2 = xx.flatten(), yy.flatten()
            r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
            # horizontal stack vectors to create x1,x2 input for the model
            grid = hstack((r1,r2))
            # define the model

            # fit the model

            # make predictions for the grid
            yhat = i.predict(pd.DataFrame(grid))
            # reshape the predictions back into a grid
            yhat=yhat.to_numpy()
            zz = yhat.reshape(xx.shape)
            # plot the grid of x, y and z values as a surface
            ax[count].contourf(xx, yy, zz, cmap='Paired',alpha=1)
            # count+=1;
            # create scatter plot for samples from each class
            y2=yt.iloc[i.data].to_numpy()
            for class_value in range(2):
            # get row indexes for samples with this class
                row_ix = where(y2 == class_value)
            # create scatter of these samples
                ax[count].scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
            count+=1;
        fig2,ax=plt.subplots()
        count=0
        for i in self.trees:
            min1, max1 = Xt.iloc[i.data, 0].min()-1, Xt.iloc[i.data, 0].max()+1
            min2, max2 = Xt.iloc[i.data, 1].min()-1, Xt.iloc[i.data, 1].max()+1
            # define the x and y scale
            x1grid = arange(min1, max1, 0.1)
            x2grid = arange(min2, max2, 0.1)
            # create all of the lines and rows of the grid
            xx, yy = meshgrid(x1grid, x2grid)
            # flatten each grid to a vector
            r1, r2 = xx.flatten(), yy.flatten()
            r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
            # horizontal stack vectors to create x1,x2 input for the model
            grid = hstack((r1,r2))
            # define the model

            # fit the model

            # make predictions for the grid
            yhat = i.predict(pd.DataFrame(grid))
            # reshape the predictions back into a grid
            yhat=yhat.to_numpy()
            zz = yhat.reshape(xx.shape)
            # plot the grid of x, y and z values as a surface
            ax.contourf(xx, yy, zz, cmap='Paired',alpha=0.3)
            # count+=1;
            # create scatter plot for samples from each class
        for class_value in range(len(list(yt.cat.categories))):
            # get row indexes for samples with this class
            row_ix = where(y == class_value)
            # create scatter of these samples
            ax.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
        return [fig1,fig2]



class RandomForestRegressor():
    trees=[]
    X_copy=pd.DataFrame()
    Y_copy=pd.Series()
    def __init__(self, n_estimators=100, criterion='variance', max_depth=1000):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.max_depth=max_depth
        self.trees=[]

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X_copy=X
        self.Y_copy=y
        for i in range(self.n_estimators):
            model=DecisionTreeRF(max_depth=self.max_depth)
            model.fitRF(X,y)
            self.trees.append(model)
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        l=[]
        for i in self.trees:
            l2=i.predict(X).tolist()
            l.append(l2)
        ans=[]
        for i in range(len(l[0])):
            summ=0
            for j in range(len(l)):
                summ+=l[j][i]
            ans.append(summ/len(l))
        return pd.Series(ans)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        fig1,ax=plt.subplots()
        X=self.X_copy.to_numpy()
        Xt=self.X_copy
        yt=self.Y_copy
        # model=RandomForestRegressor(n_estimators=10)
        # self.fit(self.X_copy,,pd.Series(y))
        min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
        min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
        # define the x and y scale
        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)
        # x3grid=np.empty(len(x1grid))

        # create all of the lines and rows of the grid
        xx, yy = np.meshgrid(x1grid, x2grid)
        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2= r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        randomm=[i for i in range(0,len(Xt))]
        random.shuffle(randomm)
        # X=Xt.iloc[randomm,:]
        # Y=yt.iloc[randomm]
        # for i in range(2,len(X[0])):
        #     x3grid=np.empty((len(r1),1))
        #     x3grid.fill(X[:,2].mean())
        # horizontal stack vectors to create x1,x2 input for the model
        grid2=np.hstack((r1,r2))
    
        grid = np.hstack((r1,r2))
        yhat=self.predict(pd.DataFrame(grid2))
        yhat=yhat.to_numpy()
        zz=yhat.reshape(xx.shape)
        cf=ax.contourf(xx,yy,zz)
        plt.colorbar(cf)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Regression Decision Surface")
        return fig1
        pass
