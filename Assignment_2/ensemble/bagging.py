from tree.base import *
from multiprocessing import Pool
class BaggingClassifier():
    trees=[]
    X_copy=pd.DataFrame()
    Y_copy=pd.Series()
    Xdata=[]
    ydata=[]
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.trees=[]
        self.X_copy=pd.DataFrame()
        self.Y_copy=pd.Series()
        self.Xdata=[]
        self.ydata=[]

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X_copy=X
        self.Y_copy=y
        for iter in range(self.n_estimators):
            lbag=set()
            # indices=[i for i in range(len(X))]
            for i in range(int(len(X)/(2.5))):
                lbag.add(random.randint(0,len(X)-1))
            lbag2=list(lbag)
            Xt=X.iloc[lbag2]
            yt=y.iloc[lbag2]
            self.Xdata.append(Xt)
            self.ydata.append(yt)
            model=DecisionTree()
            model.fit(Xt,yt)
            self.trees.append(model)

        pass
    def onetree(self,X,y):
        lbag=set()
            # indices=[i for i in range(len(X))]
        # X=self.X_copy
        # y=self.Y_copy
        for i in range(int(len(X)/(2.5))):
            lbag.add(random.randint(0,len(X)-1))
        lbag2=list(lbag)
        Xt=X.iloc[lbag2]
        yt=y.iloc[lbag2]
        model=DecisionTree()
        model.fit(Xt,yt)
        # print(1)
        return model
        # self.trees.append(model)
    def fitparallel(self,X,y):
        self.X_copy=X
        self.Y_copy=y
        # if __name__=="__main__":
        with Pool() as pool: 
            items=[(X,y) for i in range(self.n_estimators)]
            for result in pool.starmap(self.onetree,items):
                self.trees.append(result)
        # print(len(self.trees))


    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
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
                    d[l[j][i]]=1
            ans.append(max(zip(d.values(), d.keys()))[1])
        return pd.Series(ans,dtype='category')
        pass

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        
        fig1,ax=plt.subplots(1, self.n_estimators, sharex='col', sharey='row',figsize=(6*(self.n_estimators),6))
        # define the model
        for i in range(self.n_estimators):
            model = self.trees[i]
            # fit the model
            Xt=model.X_copy
            yt=model.Y_copy
            # temp=1/(len(Xt))
            # make predictions for the grid
            X=Xt.to_numpy()
            y=yt.to_numpy()
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
            yhat = model.predict(pd.DataFrame(grid))
            yhat=yhat.to_numpy()
            # reshape the predictions back into a grid
            zz = yhat.reshape(xx.shape)
            ax[i].set_title(label=f'iteration = {i+1}',y=-0.15)
            ax[i].contourf(xx,yy,zz,cmap='Paired')
            for cv in list(yt.cat.categories):
                row_ix=np.where(y==cv)
                # print(row_ix)
                ax[i].scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
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
        fig2,ax2=plt.subplots()
        yhat = self.predict(pd.DataFrame(grid))
        yhat=yhat.to_numpy()
        # reshape the predictions back into a grid
        zz = yhat.reshape(xx.shape)
        ax2.set_title(label='Combined Model',y=-0.15)
        ax2.contourf(xx,yy,zz,cmap='Paired')
        for cv in list(yt.cat.categories):
            row_ix=np.where(y==cv)
            # print(row_ix)
            ax2.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
        return [fig1,fig2]